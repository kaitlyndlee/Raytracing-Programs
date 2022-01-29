// common gdt helper tools
#include "optix7.h"
#include <iostream>


  /*! helper function that initializes optix and checks for errors */
  void initOptix()
  {
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    optixInit();
  }

  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
int main(int ac, char **av)
  {

      std::cout << "#osc: initializing optix..." << std::endl;
      
      initOptix();
       
      std::cout << "#osc: successfully initialized optix... yay!"<< std::endl;

      // for this simple hello-world example, don't do anything else
      // ...
      std::cout << "#osc: done. clean exit." << std::endl;
    return 0;
  }
  