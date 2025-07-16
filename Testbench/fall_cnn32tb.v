`timescale 1ns / 1ps
module cnn_top_tb;
    // Testbench signals
    reg clk;
    reg rst_n;
    reg start;
    reg [8191:0] input_data; // 32x32x8-bit packed array (32*32*8 = 8192 bits)
    wire signed [31:0] output_class0;
    wire signed [31:0] output_class1;
    wire done;
    wire fall;
    
    // Memory to store input image
    reg [7:0] input_image_mem [0:1023]; // 32x32 = 1024 pixels
    
    // Instantiate the CNN module
    cnn_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .input_data(input_data),
        .output_class0(output_class0),
        .output_class1(output_class1),
        .done(done),
        .fall(fall)
    );
    
    // Clock generation - 10ns period (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Task to pack image data into input_data
    task pack_input_data;
        integer i;
        begin
            input_data = 0;
            for (i = 0; i < 1024; i = i + 1) begin // Changed 4096 to 1024
                input_data[(i*8) +: 8] = input_image_mem[i];
            end
        end
    endtask
    
    // Task to display classification result
    task display_result;
        begin
            $display("================================");
            $display("CNN Fall Detection Results:");
            $display("================================");
            $display("Raw Output Class 0 (Not Fall): %d (0x%h)", output_class0, output_class0);
            $display("Raw Output Class 1 (Fall):      %d (0x%h)", output_class1, output_class1);
            $display("--------------------------------");
            
            // Simple classification based on which output is larger
            if (output_class0 > output_class1) begin
                $display("PREDICTION: NOT FALL");
                $display("Confidence Difference: %d", output_class0 - output_class1);
            end else if (output_class1 > output_class0) begin
                $display("PREDICTION: FALL DETECTED");
                $display("Confidence Difference: %d", output_class1 - output_class0);
            end else begin
                $display("PREDICTION: UNCERTAIN (Equal outputs)");
            end
            $display("================================");
        end
    endtask
    
    // Task to initialize test image with pattern (for testing without external file)
    task initialize_test_pattern;
        integer row, col, pixel_idx;
        begin
            $display("Initializing test pattern...");
            for (row = 0; row < 32; row = row + 1) begin // Changed 64 to 32
                for (col = 0; col < 32; col = col + 1) begin // Changed 64 to 32
                    pixel_idx = row * 32 + col; // Changed 64 to 32
                    // Create a simple gradient pattern for testing
                    input_image_mem[pixel_idx] = (row + col) % 256;
                end
            end
        end
    endtask
    
    // Main test sequence
    initial begin
        $display("Starting CNN Fall Detection Test (32x32)..."); // Changed 64x64 to 32x32
        
        // Initialize signals
        rst_n = 0;
        start = 0;
        input_data = 0;
        
        // Load your input image from .mem file
        // Replace "IMG20250528114349.mem" with your actual 32x32 input image file name
        $display("Loading input image from IMG20250528114349_32x32.mem..."); // Renamed for clarity
        
        // Try to load from file, if it fails, use test pattern
        if ($fopen("fall01.mem", "r") != 0) begin // Renamed for clarity
            $readmemh("fall01.mem", input_image_mem); // Renamed for clarity
            $display("Successfully loaded image from file");
        end else begin
            $display("Warning: IMG20250528114349_32x32.mem not found, using test pattern"); // Renamed for clarity
            initialize_test_pattern();
        end
        
        // Check if image loaded successfully
        $display("First few pixel values: %h %h %h %h", 
                 input_image_mem[0], input_image_mem[1], 
                 input_image_mem[2], input_image_mem[3]);
        $display("Last few pixel values: %h %h %h %h", 
                 input_image_mem[1020], input_image_mem[1021], // Changed 4092 to 1020 for 32x32
                 input_image_mem[1022], input_image_mem[1023]); // Changed 4095 to 1023 for 32x32
        
        // Pack the image data
        pack_input_data();
        
        // Reset sequence
        #20 rst_n = 1;
        $display("Reset released, starting CNN processing...");
        
        // Start processing
        #10 start = 1;
        #10 start = 0;
        
        $display("Processing started, waiting for completion...");
        $display("This may take a while for 32x32 processing...");
        
        // Wait for processing to complete
        wait(done);
        
        $display("Processing completed at time %0t", $time);
        
        // Display results
        display_result();
        
        // Additional timing
        #100;
        
        $display("Test completed successfully!");
        $finish;
    end
    
    // Monitor processing stages (optional - for debugging)
    initial begin
        $display("State monitoring enabled...");
        forever begin
            @(posedge clk);
            if (dut.state != 0) begin // Only display when not in IDLE
                case (dut.state)
                    4'b0001: $display("Time: %0t | State: LOAD_INPUT | Counter: %0d", $time, dut.input_counter);
                    4'b0010: $display("Time: %0t | State: CONV1 | Row: %0d, Col: %0d, Filter: %0d", $time, dut.conv_row, dut.conv_col, dut.conv_filter);
                    4'b0011: $display("Time: %0t | State: POOL1 | Row: %0d, Col: %0d, Filter: %0d", $time, dut.pool_row, dut.pool_col, dut.pool_filter);
                    4'b0100: $display("Time: %0t | State: CONV2 | Row: %0d, Col: %0d, Filter: %0d", $time, dut.conv_row, dut.conv_col, dut.conv_filter);
                    4'b0101: $display("Time: %0t | State: POOL2 | Row: %0d, Col: %0d, Filter: %0d", $time, dut.pool_row, dut.pool_col, dut.pool_filter);
                    4'b0110: $display("Time: %0t | State: FLATTEN | Index: %0d", $time, dut.flatten_idx);
                    4'b0111: $display("Time: %0t | State: DENSE1 | Output Index: %0d", $time, dut.dense_output_idx);
                    4'b1000: $display("Time: %0t | State: DENSE2 | Output Index: %0d", $time, dut.final_dense_idx);
                    4'b1001: $display("Time: %0t | State: OUTPUT | Done: %b", $time, done);
                    default: $display("Time: %0t | State: UNKNOWN (%b)", $time, dut.state);
                endcase
            end
        end
    end
    
    // Progress indicator for long processing
    initial begin
        #1000;
        forever begin
            #10000; // Every 10us
            if (!done) begin
                $display("Processing... Current state: %0d, Time elapsed: %0t", dut.state, $time);
            end else begin
                $finish;
            end
        end
    end
    
    // Timeout watchdog (prevents infinite simulation)
    // Adjusted timeout for 32x32 processing (will be much faster than 64x64)
    initial begin
        #5000000; // 5ms timeout (reduced from 50ms for 32x32)
        $display("ERROR: Simulation timeout! CNN processing took too long.");
        $display("Current state: %0d", dut.state);
        $display("Check if the design is functioning correctly.");
        $finish;
    end
    
    // File dump for debugging (optional)
    initial begin
        $dumpfile("cnn_tb.vcd");
        $dumpvars(0, cnn_top_tb);
    end
    
endmodule