`timescale 1ns / 1ps
module cnn_top (
    input wire clk,
    input wire rst_n,
    input wire start,
    // input_data for 32x32 grayscale image (flattened, 32*32*8 = 8192 bits)
    input wire [8191:0] input_data, 
    output reg signed [31:0] output_class0,
    output reg signed [31:0] output_class1,
    output reg done,
    output reg fall
);

    // Parameters for fixed-point arithmetic
    parameter INPUT_WIDTH = 8;
    parameter WEIGHT_WIDTH = 8;
    parameter BIAS_WIDTH = 16;
    parameter RESULT_WIDTH = 32;
    parameter FRAC_BITS = 8; // Number of fractional bits for fixed-point

    // State machine states
    parameter IDLE = 4'b0000;
    parameter LOAD_INPUT = 4'b0001;
    parameter CONV1 = 4'b0010;
    parameter POOL1 = 4'b0011;
    parameter CONV2 = 4'b0100;
    parameter POOL2 = 4'b0101;
    parameter FLATTEN = 4'b0110;
    parameter DENSE1 = 4'b0111;
    parameter DENSE2 = 4'b1000;
    parameter OUTPUT = 4'b1001;

    reg [3:0] state, next_state;

    // Memory declarations for weights and biases
    // Layer 0: Conv2D - input(32,32,1) -> output(30,30,32)
    reg signed [WEIGHT_WIDTH-1:0] conv0_weights [0:287];    // 3*3*1*32 = 288 weights
    reg signed [BIAS_WIDTH-1:0] conv0_biases [0:31];        // 32 biases

    // Layer 2: Conv2D_1 - input(15,15,32) -> output(13,13,64)
    reg signed [WEIGHT_WIDTH-1:0] conv1_weights [0:18431]; // 3*3*32*64 = 18432 weights
    reg signed [BIAS_WIDTH-1:0] conv1_biases [0:63];        // 64 biases

    // Layer 5: Dense - input(2304) -> output(128)
    reg signed [WEIGHT_WIDTH-1:0] dense0_weights [0:294911]; // 2304*128 = 294,912 weights
    reg signed [BIAS_WIDTH-1:0] dense0_biases [0:127];     // 128 biases

    // Layer 7: Dense_1 - input(128) -> output(2)
    reg signed [WEIGHT_WIDTH-1:0] dense1_weights [0:255];   // 128*2 = 256 weights
    reg signed [BIAS_WIDTH-1:0] dense1_biases [0:1];        // 2 biases

    // Internal arrays for data flow - using proper bit widths
    reg signed [INPUT_WIDTH-1:0] input_image [0:31][0:31];    // 32x32 input
    reg signed [RESULT_WIDTH-1:0] conv0_output [0:29][0:29][0:31];     // 30x30x32
    reg signed [RESULT_WIDTH-1:0] pool0_output [0:14][0:14][0:31];     // 15x15x32
    reg signed [RESULT_WIDTH-1:0] conv1_output [0:12][0:12][0:63];     // 13x13x64
    reg signed [RESULT_WIDTH-1:0] pool1_output [0:5][0:5][0:63];      // 6x6x64
    reg signed [RESULT_WIDTH-1:0] flatten_output [0:2303];            // 2304 elements (6*6*64)
    reg signed [RESULT_WIDTH-1:0] dense0_output [0:127];             // 128 outputs
    reg signed [RESULT_WIDTH-1:0] dense1_output [0:1];               // 2 outputs

    // Control signals and counters
    reg [9:0] input_counter;        // 0 to 1023 (32*32-1)
    reg [4:0] conv_row, conv_col;   // 0 to 29 for first conv, 0 to 12 for second
    reg [5:0] conv_filter;          // 0 to 31 for first conv, 0 to 63 for second
    reg [4:0] pool_row, pool_col;   // Pool indices (0 to 14 for first, 0 to 5 for second)
    reg [5:0] pool_filter;          // Pool filter index
    reg [11:0] dense_input_idx;    // 0 to 2303
    reg [6:0] dense_output_idx;     // 0 to 127
    reg [11:0] flatten_idx;          // Flatten index
    reg [1:0] final_dense_idx;      // Final dense layer index

    // Working variables with proper bit widths
    reg signed [RESULT_WIDTH+8-1:0] conv_sum;      // Extra bits for accumulation
    reg signed [RESULT_WIDTH-1:0] max_val;
    reg signed [RESULT_WIDTH-1:0] temp_vals [0:3];
    reg signed [RESULT_WIDTH+16-1:0] dense_sum;    // Extra bits for large accumulation

    // Loop variables
    integer kr, kc, kf, w_idx;
    integer flat_row, flat_col, flat_filter;

    // input_data extraction temporary variable (now directly signed)
    reg signed [INPUT_WIDTH-1:0] current_input_pixel;

    // Initialize memory files
    initial begin
        $readmemh("layer_0_conv2d_weights.mem", conv0_weights);
        $readmemh("layer_0_conv2d_biases.mem", conv0_biases);
        $readmemh("layer_2_conv2d_1_weights.mem", conv1_weights);
        $readmemh("layer_2_conv2d_1_biases.mem", conv1_biases);
        $readmemh("layer_5_dense_weights.mem", dense0_weights);
        $readmemh("layer_5_dense_biases.mem", dense0_biases);
        $readmemh("layer_7_dense_1_weights.mem", dense1_weights);
        $readmemh("layer_7_dense_1_biases.mem", dense1_biases);
    end

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    // Next state logic
    always @(*) begin
        case (state)
            IDLE: begin
                if (start)
                    next_state = LOAD_INPUT;
                else
                    next_state = IDLE;
            end
            LOAD_INPUT: begin
                if (input_counter == 1023) // 32*32-1
                    next_state = CONV1;
                else
                    next_state = LOAD_INPUT;
            end
            CONV1: begin
                if (conv_row == 29 && conv_col == 29 && conv_filter == 31)
                    next_state = POOL1;
                else
                    next_state = CONV1;
            end
            POOL1: begin
                if (pool_row == 14 && pool_col == 14 && pool_filter == 31)
                    next_state = CONV2;
                else
                    next_state = POOL1;
            end
            CONV2: begin
                if (conv_row == 12 && conv_col == 12 && conv_filter == 63)
                    next_state = POOL2;
                else
                    next_state = CONV2;
            end
            POOL2: begin
                if (pool_row == 5 && pool_col == 5 && pool_filter == 63)
                    next_state = FLATTEN;
                else
                    next_state = POOL2;
            end
            FLATTEN: begin
                if (flatten_idx == 2303) // 6*6*64-1
                    next_state = DENSE1;
                else
                    next_state = FLATTEN;
            end
            DENSE1: begin
                if (dense_output_idx == 127)
                    next_state = DENSE2;
                else
                    next_state = DENSE1;
            end
            DENSE2: begin
                if (final_dense_idx == 1)
                    next_state = OUTPUT;
                else
                    next_state = DENSE2;
            end
            OUTPUT: begin
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    // Main processing logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all counters and outputs
            input_counter <= 0;
            conv_row <= 0;
            conv_col <= 0;
            conv_filter <= 0;
            pool_row <= 0;
            pool_col <= 0;
            pool_filter <= 0;
            dense_input_idx <= 0;
            dense_output_idx <= 0;
            flatten_idx <= 0;
            final_dense_idx <= 0;
            output_class0 <= 0;
            output_class1 <= 0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    input_counter <= 0;
                    conv_row <= 0;
                    conv_col <= 0;
                    conv_filter <= 0;
                    pool_row <= 0;
                    pool_col <= 0;
                    pool_filter <= 0;
                    dense_input_idx <= 0;
                    dense_output_idx <= 0;
                    flatten_idx <= 0;
                    final_dense_idx <= 0;
                end

                LOAD_INPUT: begin
                    // Load and assign input data directly, as it's now expected to be signed.
                    // The slicing input_data[(input_counter * 8) +: 8] will extract 8 bits.
                    current_input_pixel = input_data[(input_counter * 8) +: 8];
                    input_image[input_counter / 32][input_counter % 32] <= current_input_pixel; // Changed /64 and %64 to /32 and %32

                    if (input_counter < 1023) // 32*32-1
                        input_counter <= input_counter + 1;
                end

                CONV1: begin
                    // First convolution layer: 32x32x1 -> 30x30x32
                    conv_sum = {{(RESULT_WIDTH+8-BIAS_WIDTH){conv0_biases[conv_filter][BIAS_WIDTH-1]}}, conv0_biases[conv_filter]};

                    for (kr = 0; kr < 3; kr = kr + 1) begin
                        for (kc = 0; kc < 3; kc = kc + 1) begin
                            conv_sum = conv_sum +
                                ($signed(input_image[conv_row + kr][conv_col + kc]) *
                                 $signed(conv0_weights[conv_filter * 9 + kr * 3 + kc]));
                        end
                    end

                    // ReLU activation with proper scaling
                    if (conv_sum[RESULT_WIDTH+8-1] == 1'b1) // negative
                        conv0_output[conv_row][conv_col][conv_filter] <= 0;
                    else
                        conv0_output[conv_row][conv_col][conv_filter] <= conv_sum[RESULT_WIDTH-1:0];

                    // Update counters
                    if (conv_filter == 31) begin
                        conv_filter <= 0;
                        if (conv_col == 29) begin // Changed 61 to 29
                            conv_col <= 0;
                            if (conv_row == 29) begin // Changed 61 to 29
                                conv_row <= 0;
                            end else begin
                                conv_row <= conv_row + 1;
                            end
                        end else begin
                            conv_col <= conv_col + 1;
                        end
                    end else begin
                        conv_filter <= conv_filter + 1;
                    end
                end

                POOL1: begin
                    // First max pooling: 30x30x32 -> 15x15x32
                    temp_vals[0] = conv0_output[pool_row * 2][pool_col * 2][pool_filter];
                    temp_vals[1] = conv0_output[pool_row * 2][pool_col * 2 + 1][pool_filter];
                    temp_vals[2] = conv0_output[pool_row * 2 + 1][pool_col * 2][pool_filter];
                    temp_vals[3] = conv0_output[pool_row * 2 + 1][pool_col * 2 + 1][pool_filter];

                    // Find maximum using proper signed comparison
                    max_val = temp_vals[0];
                    if ($signed(temp_vals[1]) > $signed(max_val)) max_val = temp_vals[1];
                    if ($signed(temp_vals[2]) > $signed(max_val)) max_val = temp_vals[2];
                    if ($signed(temp_vals[3]) > $signed(max_val)) max_val = temp_vals[3];

                    pool0_output[pool_row][pool_col][pool_filter] <= max_val;

                    // Update counters
                    if (pool_filter == 31) begin
                        pool_filter <= 0;
                        if (pool_col == 14) begin // Changed 30 to 14
                            pool_col <= 0;
                            if (pool_row == 14) begin // Changed 30 to 14
                                pool_row <= 0;
                                conv_row <= 0; // Reset for next stage
                                conv_col <= 0;
                                conv_filter <= 0;
                            end else begin
                                pool_row <= pool_row + 1;
                            end
                        end else begin
                            pool_col <= pool_col + 1;
                        end
                    end else begin
                        pool_filter <= pool_filter + 1;
                    end
                end

                CONV2: begin
                    // Second convolution layer: 15x15x32 -> 13x13x64
                    conv_sum = {{(RESULT_WIDTH+8-BIAS_WIDTH){conv1_biases[conv_filter][BIAS_WIDTH-1]}}, conv1_biases[conv_filter]};

                    for (kr = 0; kr < 3; kr = kr + 1) begin
                        for (kc = 0; kc < 3; kc = kc + 1) begin
                            for (kf = 0; kf < 32; kf = kf + 1) begin
                                conv_sum = conv_sum +
                                    ($signed(pool0_output[conv_row + kr][conv_col + kc][kf]) *
                                     $signed(conv1_weights[conv_filter * 288 + kr * 96 + kc * 32 + kf])); // 3*32*64 = 18432. The index for weights is correct based on original
                            end
                        end
                    end

                    // ReLU activation
                    if (conv_sum[RESULT_WIDTH+8-1] == 1'b1) // negative
                        conv1_output[conv_row][conv_col][conv_filter] <= 0;
                    else
                        conv1_output[conv_row][conv_col][conv_filter] <= conv_sum[RESULT_WIDTH-1:0];

                    // Update counters
                    if (conv_filter == 63) begin
                        conv_filter <= 0;
                        if (conv_col == 12) begin // Changed 28 to 12
                            conv_col <= 0;
                            if (conv_row == 12) begin // Changed 28 to 12
                                conv_row <= 0;
                                pool_row <= 0; // Reset for next stage
                                pool_col <= 0;
                                pool_filter <= 0;
                            end else begin
                                conv_row <= conv_row + 1;
                            end
                        end else begin
                            conv_col <= conv_col + 1;
                        end
                    end else begin
                        conv_filter <= conv_filter + 1;
                    end
                end

                POOL2: begin
                    // Second max pooling: 13x13x64 -> 6x6x64
                    temp_vals[0] = conv1_output[pool_row * 2][pool_col * 2][pool_filter];
                    temp_vals[1] = conv1_output[pool_row * 2][pool_col * 2 + 1][pool_filter];
                    temp_vals[2] = conv1_output[pool_row * 2 + 1][pool_col * 2][pool_filter];
                    temp_vals[3] = conv1_output[pool_row * 2 + 1][pool_col * 2 + 1][pool_filter];

                    // Find maximum using proper signed comparison
                    max_val = temp_vals[0];
                    if ($signed(temp_vals[1]) > $signed(max_val)) max_val = temp_vals[1];
                    if ($signed(temp_vals[2]) > $signed(max_val)) max_val = temp_vals[2];
                    if ($signed(temp_vals[3]) > $signed(max_val)) max_val = temp_vals[3];

                    pool1_output[pool_row][pool_col][pool_filter] <= max_val;

                    // Update counters
                    if (pool_filter == 63) begin
                        pool_filter <= 0;
                        if (pool_col == 5) begin // Changed 13 to 5
                            pool_col <= 0;
                            if (pool_row == 5) begin // Changed 13 to 5
                                pool_row <= 0;
                                flatten_idx <= 0; // Reset for next stage
                            end else begin
                                pool_row <= pool_row + 1;
                            end
                        end else begin
                            pool_col <= pool_col + 1;
                        end
                    end else begin
                        pool_filter <= pool_filter + 1;
                    end
                end

                FLATTEN: begin
                    // Flatten 6x6x64 to 2304
                    flat_row = flatten_idx / (6 * 64);      // 6 * 64 = 384
                    flat_col = (flatten_idx % (6 * 64)) / 64;
                    flat_filter = flatten_idx % 64;

                    flatten_output[flatten_idx] <= pool1_output[flat_row][flat_col][flat_filter];

                    if (flatten_idx < 2303) // Changed 12543 to 2303
                        flatten_idx <= flatten_idx + 1;
                end

                DENSE1: begin
                    // First dense layer: 2304 -> 128
                    dense_sum = {{(RESULT_WIDTH+16-BIAS_WIDTH){dense0_biases[dense_output_idx][BIAS_WIDTH-1]}}, dense0_biases[dense_output_idx]};

                    for (w_idx = 0; w_idx < 2304; w_idx = w_idx + 1) begin // Changed 12544 to 2304
                        dense_sum = dense_sum +
                            ($signed(flatten_output[w_idx]) *
                             $signed(dense0_weights[dense_output_idx * 2304 + w_idx])); // Changed 12544 to 2304
                    end

                    // ReLU activation with proper bit selection
                    if (dense_sum[RESULT_WIDTH+16-1] == 1'b1) // negative
                        dense0_output[dense_output_idx] <= 0;
                    else
                        dense0_output[dense_output_idx] <= dense_sum[RESULT_WIDTH-1:0];

                    if (dense_output_idx < 127)
                        dense_output_idx <= dense_output_idx + 1;
                end

                DENSE2: begin
                    // Second dense layer (output): 128 -> 2
                    dense_sum = {{(RESULT_WIDTH+16-BIAS_WIDTH){dense1_biases[final_dense_idx][BIAS_WIDTH-1]}}, dense1_biases[final_dense_idx]};

                    for (w_idx = 0; w_idx < 128; w_idx = w_idx + 1) begin
                        dense_sum = dense_sum +
                            ($signed(dense0_output[w_idx]) *
                             $signed(dense1_weights[final_dense_idx * 128 + w_idx]));
                    end

                    // No activation function for output layer (linear)
                    dense1_output[final_dense_idx] <= dense_sum[RESULT_WIDTH-1:0];

                    if (final_dense_idx < 1)
                        final_dense_idx <= final_dense_idx + 1;
                end

                OUTPUT: begin
                    // Output final results
                    output_class0 <= dense1_output[0];
                    output_class1 <= dense1_output[1];
                    done <= 1;
                if (dense1_output[1] > dense1_output[0])                        
                        fall <= 1; // Fall detected
                    else
                        fall <= 0; // Not a fall
                end
            endcase
        end
    end

endmodule