// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "3rdParty/stb_image_write.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    try {
      SampleRenderer sample;

      int width = 1200;
      int height = 1024;
      sample.resize(width, height);
      sample.render();

      // std::vector<uint8_t> pixels(width * height * 3);
      // sample.downloadPixels(pixels.data());

      FILE *output_image = fopen("tutorial.ppm", "wb");
      if(output_image == NULL) {
        fprintf(stderr, "Error: Unable to open the output image file: %s\n", "tutorial.ppm");
        exit(1);
      }
      sample.writeToFile(output_image);

    } catch (std::runtime_error& e) {
      std::cout << "FATAL ERROR: " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }
  
} // ::osc
