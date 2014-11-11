#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

      // load opencl source
      std::ifstream cl_file("convolution.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
      program.build(devices);
       
       // initialize data
       std::vector<float> matrixOne;
       std::vector<float> matrixTwo;
       std::vector<float> matrixOut;
       
       std::ifstream istream("input.txt");
       std::ofstream ostream("output.txt");
       int sizeFirst = 0;
       int sizeMask = 0;
       istream >> sizeFirst >> sizeMask;
    
      matrixOne.resize(sizeFirst * sizeFirst);
       for (int i = 0; i < sizeFirst; ++i) {
           for (int j = 0; j < sizeFirst; ++j) {
               istream >> matrixOne[i * sizeFirst + j];
           }
       }
       matrixTwo.resize(sizeMask * sizeMask);
       for (int i = 0; i < sizeMask; ++i) {
           for (int j = 0; j < sizeMask; ++j) {
            istream >> matrixTwo[i * sizeMask + j];
           }
       }

      // create a message to send to kernel
      size_t const block_size = 8;

      // allocate device buffer to hold message
      cl::Buffer dev_first (context, CL_MEM_READ_ONLY, sizeof(float) * sizeFirst * sizeFirst);
       cl::Buffer dev_mask (context, CL_MEM_READ_ONLY, sizeof(float) * sizeMask * sizeMask);
       cl::Buffer dev_out(context, CL_MEM_WRITE_ONLY, sizeof(float) * sizeFirst * sizeFirst);


      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_first, CL_TRUE, 0, sizeof(float) * sizeFirst * sizeFirst, &matrixOne[0]);
      queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(float)* sizeMask * sizeMask, &matrixTwo[0]);

      // load named kernel from opencl source
      queue.finish();
       
       
        // in order to divide on whole number of full blocks - round up to
       // nearest modulo(block_size) number
       
        size_t global_size = ((sizeFirst + block_size - 1) / block_size) * block_size;

       
       cl::Kernel kernel(program, "convolution");
       cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(global_size, global_size), cl::NDRange(block_size, block_size));
       cl::Event event = convolution(dev_first, dev_mask, dev_out, sizeFirst, sizeMask);
     
      event.wait();
       
      cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
      cl_ulong elapsed_time = end_time - start_time;
       
      matrixOut.resize(sizeFirst * sizeFirst);
      queue.enqueueReadBuffer(dev_out, CL_TRUE, 0, sizeof(float) * sizeFirst * sizeFirst, &matrixOut[0]);
       
       ostream << std::fixed << std::setprecision(3);
       
      for (size_t i = 0; i < sizeFirst; ++i) {
          for (int j = 0; j < sizeFirst; ++j) {
              ostream << matrixOut[i * sizeFirst + j] << "\t";
          }
           ostream << "\n";
      }

    std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}