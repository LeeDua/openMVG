// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DISTRIBUTED_BA_DATA_HPP
#define DISTRIBUTED_BA_DATA_HPP

#include "openMVG/cameras/Camera_Common.hpp"

namespace openMVG {
  namespace sfm {

    struct Global_Params{
      int total_intrinsics;
      int total_poses;
      int total_observations;
      int total_tracks;
      explicit Global_Params(int total_intrinsics,int total_poses, int total_observations, int total_tracks)
        :total_intrinsics(total_intrinsics),
         total_poses(total_poses),
         total_observations(total_observations),
         total_tracks(total_tracks){};
    };

    struct Extrinsic_Coes{
      double rho_r;
      double rho_c;
      explicit Extrinsic_Coes(double rho_r,double rho_c):rho_r(rho_r), rho_c(rho_c){};
    };

    struct Intrinsic_Coes{
      double rho_f,rho_uv,rho_d;
      explicit Intrinsic_Coes(double rho_f,double rho_uv,double rho_d):rho_f(rho_f),rho_uv(rho_uv),rho_d(rho_d){};
    };


  } // namespace sfm
} // namespace openMVG

#endif // DISTRIBUTED_BA_DATA_HPP
