/*! \file

\verbatim

Copyright (c) 2004, Sylvain Paris and Francois Sillion
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    
    * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    * Neither the name of ARTIS, GRAVIR-IMAG nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

\endverbatim


 *  This file contains code made by Sylvain Paris under supervision of
 * François Sillion for his PhD work with <a
 * href="http://www-artis.imag.fr">ARTIS project</a>. ARTIS is a
 * research project in the GRAVIR/IMAG laboratory, a joint unit of
 * CNRS, INPG, INRIA and UJF.
 *
 *  Use <a href="http://www.stack.nl/~dimitri/doxygen/">Doxygen</a>
 * with DISTRIBUTE_GROUP_DOC option to produce an nice html
 * documentation.
 *
 *  The file defines a class to handle layered images.
 */

#ifndef __CHANNEL_IMAGE__
#define __CHANNEL_IMAGE__

#include "array.h"

enum channel_meaning_type {LAB,LAB_RATIO,RGB,RGBA,SCALAR,NOT_DEFINED};

/*! This class represents an image composed of several channels.*/
template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
class Channel_image{

public:
  
  typedef Real         real_type;
  typedef unsigned int size_type;
  
  typedef Array_2D<real_type> channel_type;

  
  Channel_image();
  Channel_image(const Channel_image<N,Real,Channel_meaning>& c);
  Channel_image(const size_type x,const size_type y);

  void resize(const size_type x,const size_type y);

  size_type size() const;
  size_type x_size() const;
  size_type width() const;
  size_type y_size() const;
  size_type height() const;
  size_type channel_size() const;
  channel_meaning_type channel_meaning() const;

  channel_type& operator[](const size_type c);
  const channel_type& operator[](const size_type c) const;
  
private:
  Array_2D<real_type> channel[N];
  
};


template<typename Real>
class RGB_channel_image:public Channel_image<3,Real,RGB>{

public:

  typedef Real                                          real_type;
  typedef typename Channel_image<3,Real,RGB>::size_type size_type;
  
  static const size_type RED   = 0;
  static const size_type GREEN = 1;
  static const size_type BLUE  = 2;

  RGB_channel_image()
    :Channel_image<3,Real,RGB>(){};

  RGB_channel_image(const RGB_channel_image<Real>& c)
    :Channel_image<3,Real,RGB>(c){}
  
  RGB_channel_image(const size_type x,const size_type y)
    :Channel_image<3,Real,RGB>(x,y){}  
};


typedef RGB_channel_image<float>  RGB_f_channel_image;
typedef RGB_channel_image<double> RGB_d_channel_image;



template<typename Real>
class LAB_channel_image:public Channel_image<3,Real,LAB>{

public:

  typedef Real                                          real_type;
  typedef typename Channel_image<3,Real,LAB>::size_type size_type;
  
  static const size_type L = 0;
  static const size_type A = 1;
  static const size_type B = 2;

  LAB_channel_image()
    :Channel_image<3,Real,LAB>(){};

  LAB_channel_image(const LAB_channel_image<Real>& c)
    :Channel_image<3,Real,LAB>(c){}
  
  LAB_channel_image(const size_type x,const size_type y)
    :Channel_image<3,Real,LAB>(x,y){}  
};


typedef LAB_channel_image<float>  LAB_f_channel_image;
typedef LAB_channel_image<double> LAB_d_channel_image;





template<typename Real>
class LAB_RATIO_channel_image:public Channel_image<3,Real,LAB_RATIO>{

public:

  typedef Real                                                real_type;
  typedef typename Channel_image<3,Real,LAB_RATIO>::size_type size_type;
  
  static const size_type L        = 0;
  static const size_type A_OVER_L = 1;
  static const size_type B_OVER_L = 2;

  LAB_RATIO_channel_image()
    :Channel_image<3,Real,LAB_RATIO>(){};

  LAB_RATIO_channel_image(const LAB_RATIO_channel_image<Real>& c)
    :Channel_image<3,Real,LAB_RATIO>(c){}
  
  LAB_RATIO_channel_image(const size_type x,const size_type y)
    :Channel_image<3,Real,LAB_RATIO>(x,y){}  
};


typedef LAB_RATIO_channel_image<float>  LAB_RATIO_f_channel_image;
typedef LAB_RATIO_channel_image<double> LAB_RATIO_d_channel_image;






template<typename Real>
class RGBA_channel_image:public Channel_image<4,Real,RGBA>{

public:

  typedef Real                                           real_type;
  typedef typename Channel_image<4,Real,RGBA>::size_type size_type;
  
  static const size_type RED   = 0;
  static const size_type GREEN = 1;
  static const size_type BLUE  = 2;
  static const size_type ALPHA = 3;

  RGBA_channel_image()
    :Channel_image<4,Real,RGBA>(){};

  RGBA_channel_image(const RGBA_channel_image<Real>& c)
    :Channel_image<4,Real,RGBA>(c){}
  
  RGBA_channel_image(const size_type x,const size_type y)
    :Channel_image<4,Real,RGBA>(x,y){}  
};

typedef RGBA_channel_image<float>  RGBA_f_channel_image;
typedef RGBA_channel_image<double> RGBA_d_channel_image;


template<typename Real>
class Mono_channel_image:public Channel_image<1,Real,SCALAR>{

public:

  typedef Real                                             real_type;
  typedef typename Channel_image<1,Real,SCALAR>::size_type size_type;
  
  static const size_type CHANNEL = 0;

  Mono_channel_image()
    :Channel_image<1,Real,SCALAR>(){};

  Mono_channel_image(const Mono_channel_image<Real>& c)
    :Channel_image<1,Real,SCALAR>(c){}
  
  Mono_channel_image(const size_type x,const size_type y)
    :Channel_image<1,Real,SCALAR>(x,y){}  
};


typedef Mono_channel_image<float>  Mono_f_channel_image;
typedef Mono_channel_image<double> Mono_d_channel_image;


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


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
Channel_image<N,Real,Channel_meaning>::Channel_image(){}



template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
Channel_image<N,Real,Channel_meaning>::Channel_image(const Channel_image<N,Real,Channel_meaning>& c){

  for(size_type i=0;i<N;i++){
    channel[i] = c[i];
  }
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
Channel_image<N,Real,Channel_meaning>::Channel_image(const size_type x,const size_type y){

  for(size_type i=0;i<N;i++){
    channel[i].resize(x,y);
  }  
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
void Channel_image<N,Real,Channel_meaning>::resize(const size_type x,const size_type y){

  for(size_type i=0;i<N;i++){
    channel[i].resize(x,y);
  }
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::size() const{

  return channel[0].size();
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::x_size() const{

  return channel[0].x_size();
}



template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::width() const{

  return channel[0].x_size();
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::y_size() const{

  return channel[0].y_size();
}

template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::height() const{

  return channel[0].y_size();
}



template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::size_type
Channel_image<N,Real,Channel_meaning>::channel_size() const{
  
  return N;
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
channel_meaning_type
Channel_image<N,Real,Channel_meaning>::channel_meaning() const{

  return Channel_meaning;
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
typename Channel_image<N,Real,Channel_meaning>::channel_type&
Channel_image<N,Real,Channel_meaning>::operator[](const size_type c){

  return channel[c];
}


template <unsigned int N,typename Real,channel_meaning_type Channel_meaning>
const typename Channel_image<N,Real,Channel_meaning>::channel_type&
Channel_image<N,Real,Channel_meaning>::operator[](const size_type c) const{

  return channel[c];
}

#endif
