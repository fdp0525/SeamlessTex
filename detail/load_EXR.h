/*! \file
  \verbatim
  
    Copyright (c) 2006, Sylvain Paris and Frédo Durand

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

  \endverbatim
*/


#ifndef __LOAD_EXR__
#define __LOAD_EXR__


#include <algorithm>
#include <limits>

#include "channel_image.h"

// Clean the place for OpenEXR
#undef APPEND_EXC
#undef ASSERT
#undef REPLACE_EXC
#undef THROW
#undef THROW_ERRNO

#include "OpenEXR/ImfArray.h"
#include "OpenEXR/ImathBox.h"
#include "OpenEXR/ImfRgbaFile.h"

// Remove the macros from OpenEXR
#undef APPEND_EXC
#undef ASSERT
#undef REPLACE_EXC
#undef THROW
#undef THROW_ERRNO



namespace Image_file{

  namespace EXR{

    typedef RGBA_d_channel_image image_type;
    
    inline void load(const char*       file_name,
		     image_type* const target);

    inline void save(const char*       file_name,
		     const image_type& source);


    
/*
  
  #############################################
  #############################################
  #############################################
  ######                                 ######
  ######   I M P L E M E N T A T I O N   ######
  ######                                 ######
  #############################################
  #############################################
  #############################################
  
*/


    void load(const char*       file_name, image_type* const target)
    {

      using namespace std;
      
      typedef unsigned int size_type;
      typedef double       real_type;

      const real_type gamma = 2.2;

      Imf::RgbaInputFile file(file_name);

      Imath::Box2i dw = file.dataWindow();

      const int width  = dw.max.x - dw.min.x + 1;
      const int height = dw.max.y - dw.min.y + 1;

      Imf::Array2D<Imf::Rgba> pixels;
      pixels.resizeErase(height, width);

      file.setFrameBuffer (&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
      file.readPixels (dw.min.y, dw.max.y);

      const size_type uw = static_cast<size_type>(width);
      const size_type uh = static_cast<size_type>(height);
      
      target->resize(uw,uh);
	
      for(size_type x=0;x<uw;x++){
	for(size_type y=0;y<uh;y++){

	  const size_type ry = height - 1 - y; 

	  // We apply a gamma correction in order to make the RGB
	  // values gamma corrected as most of the images are.

	  (*target)[image_type::RED](x,y)   = pixels[ry][x].r;
// 	    std::pow(static_cast<double>(pixels[ry][x].r),1.0/gamma);

	  (*target)[image_type::GREEN](x,y) = pixels[ry][x].g;
// 	    std::pow(static_cast<double>(pixels[ry][x].g),1.0/gamma);
		     
	  (*target)[image_type::BLUE](x,y)  = pixels[ry][x].b;
// 	    std::pow(static_cast<double>(pixels[ry][x].b),1.0/gamma);
		     
	  (*target)[image_type::ALPHA](x,y) = pixels[ry][x].a;
	}
      }

      for(size_type c=0;c<target->channel_size();c++){

	double min_value = std::numeric_limits<double>::max();
	double max_value = -std::numeric_limits<double>::max();

	for(image_type::channel_type::iterator i = (*target)[c].begin();
	    i!=(*target)[c].end();
	    i++){

	  if (*i>0){
	    min_value = std::min(min_value,*i);
	  }

      
	  if (*i<std::numeric_limits<double>::max()){
	    max_value = std::max(max_value,*i);
	  }

	} // END OF for i
	
	for(image_type::channel_type::iterator i = (*target)[c].begin();
	    i!=(*target)[c].end();
	    i++){

	  if (!(*i>=min_value)){ // it should solve nan issue
	    *i = 0.75*min_value;
	  }
	  
	  if (!(*i<=max_value)){
	    *i = 1.5*max_value;
	  }
	} // END OF for i

      } // END OF for c

    }


    

    void save(const char*       file_name,
	      const image_type& source){

      typedef unsigned int size_type;
      typedef double       real_type;

      const size_type width  = source.x_size();
      const size_type height = source.y_size();

      const real_type gamma = 2.2;
      
      Imf::RgbaOutputFile file(file_name,width,height,Imf::WRITE_RGBA);

      Imf::Array2D<Imf::Rgba> pixels;
      pixels.resizeErase(height,width);


      for(size_type x=0;x<width;x++){
	for(size_type y=0;y<height;y++){

	  const size_type ry = height - 1 - y; 

	  // We remove the gamma correction.
	  pixels[ry][x].r = std::pow(source[image_type::RED](x,y),gamma);
	  pixels[ry][x].g = std::pow(source[image_type::GREEN](x,y),gamma);
	  pixels[ry][x].b = std::pow(source[image_type::BLUE](x,y),gamma);
	  pixels[ry][x].a = source[image_type::ALPHA](x,y);	  

	}
      }


      file.setFrameBuffer(&(pixels[0][0]),1,width);	
      file.writePixels(height);				    

    }
 

  } // END OF namespace EXR

} // END OF namespace Image_file



#endif
