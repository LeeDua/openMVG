// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DISTRIBUTED_BA_CERES_RESIDUAL_FUNCTOR
#define DISTRIBUTED_BA_CERES_RESIDUAL_FUNCTOR

#include <memory>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/cameras/Camera_Pinhole_Radial.hpp"
#include "software/distributed_ba/BA/distributed_BA_data.hpp"

//--
//- Define ceres Cost_functor for each OpenMVG camera model
//--

namespace openMVG {
  namespace sfm {


/**
 * @brief Ceres functor to use a Pinhole_Intrinsic_Radial_K3
 *
 *  Data parameter blocks are the following <2,6,6,3>
 *  - 2 => dimension of the residuals,
 *  - 6 => the intrinsic data block [focal, principal point x, principal point y, K1, K2, K3],
 *  - 6 => the camera extrinsic data block (camera orientation and position) [R;t],
 *         - rotation(angle axis), and translation [rX,rY,rZ,tx,ty,tz].
 *  - 3 => a 3D point data block.
 *
 */
    enum : uint8_t {
      OFFSET_FOCAL_LENGTH = 0,
      OFFSET_PRINCIPAL_POINT_X = 1,
      OFFSET_PRINCIPAL_POINT_Y = 2,
      OFFSET_DISTO_K1 = 3,
      OFFSET_DISTO_K2 = 4,
      OFFSET_DISTO_K3 = 5,
    };

    struct Distributed_ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3 {
      explicit Distributed_ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3(const double *const pos_2dpoint)
          : m_pos_2dpoint(pos_2dpoint) {
      }

      // Enum to map intrinsics parameters between openMVG & ceres camera data parameter block.


      /**
       * @param[in] cam_intrinsics: Camera intrinsics( focal, principal point [x,y], k1, k2, k3 )
       * @param[in] cam_extrinsics: Camera parameterized using one block of 6 parameters [R;t]:
       *   - 3 for rotation(angle axis), 3 for translation
       * @param[in] pos_3dpoint
       * @param[out] out_residuals
       */
      template<typename T>
      bool operator()(
          const T *const cam_intrinsics,
          const T *const cam_extrinsics,
          const T *const pos_3dpoint,
          T *out_residuals) const {

        Eigen::Matrix<T, 3, 3> R;
        ceres::AngleAxisToRotationMatrix((const T*)(cam_extrinsics), R.data());
        Eigen::Map<Eigen::Matrix<T, 3, 1>>  cam_c(&cam_extrinsics[3]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>>  point3d(&pos_3dpoint[0]);
        Eigen::Matrix<T, 3, 1> trans_point = R * ( point3d - cam_c);

        // Transform the point from homogeneous to euclidean (undistorted point)
        const Eigen::Matrix<T, 2, 1> projected_point = trans_point.hnormalized();

        //--
        // Apply intrinsic parameters
        //--
        const T &focal = cam_intrinsics[OFFSET_FOCAL_LENGTH];
        const T &principal_point_x = cam_intrinsics[OFFSET_PRINCIPAL_POINT_X];
        const T &principal_point_y = cam_intrinsics[OFFSET_PRINCIPAL_POINT_Y];
        const T &k1 = cam_intrinsics[OFFSET_DISTO_K1];
        const T &k2 = cam_intrinsics[OFFSET_DISTO_K2];
        const T &k3 = cam_intrinsics[OFFSET_DISTO_K3];

        // Apply distortion (xd,yd) = disto(x_u,y_u)
        const T r2 = projected_point.squaredNorm();
        const T r4 = r2 * r2;
        const T r6 = r4 * r2;
        const T r_coeff = (1.0 + k1 * r2 + k2 * r4 + k3 * r6);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(out_residuals);
        residuals << principal_point_x + (projected_point.x() * r_coeff) * focal - m_pos_2dpoint[0],
            principal_point_y + (projected_point.y() * r_coeff) * focal - m_pos_2dpoint[1];

        return true;
      }

      static int num_residuals() { return 2; }

      // Factory to hide the construction of the CostFunction object from
      // the client code.
      static ceres::CostFunction *Create
          (
              const Vec2 &observation
          ) {
        return
            (new ceres::AutoDiffCostFunction
                <Distributed_ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3, 2, 6, 6, 3>(
                new Distributed_ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3(observation.data())));

      }

      const double *m_pos_2dpoint; // The 2D observation
    };


    template<typename T>
    struct Distributed_Extrinsics_Loss{
      const Extrinsic_Coes* coes;
      const T* global_extrinsics;
      const T* latent_extrinsics;

      explicit Distributed_Extrinsics_Loss(const Extrinsic_Coes* coes, const T* global_extrinsics, const T* latent_extrinsics):
          coes(coes),global_extrinsics(global_extrinsics),latent_extrinsics(latent_extrinsics){};

      bool operator()(
          const T*const extrinsics,
          T* out_residuals
      )const{
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(out_residuals) << (extrinsics[0] - global_extrinsics[0] + latent_extrinsics[0]) * sqrt(coes->rho_r),
           (extrinsics[1] - global_extrinsics[1] + latent_extrinsics[1]) * sqrt(coes->rho_r),
           (extrinsics[2] - global_extrinsics[2] + latent_extrinsics[2]) * sqrt(coes->rho_r),
           (extrinsics[3] - global_extrinsics[3] + latent_extrinsics[3]) * sqrt(coes->rho_c),
           (extrinsics[4] - global_extrinsics[4] + latent_extrinsics[4]) * sqrt(coes->rho_c),
           (extrinsics[5] - global_extrinsics[5] + latent_extrinsics[5]) * sqrt(coes->rho_c);
        return true;
      }

      static int num_residuals(){ return 6; }

      static ceres::CostFunction *Create
          (
              const Extrinsic_Coes* coes,
              const T* global_extrinsics,
              const T* latent_extrinsics
          ){
        return
            (new ceres::AutoDiffCostFunction
                <Distributed_Extrinsics_Loss,6,6>(
                new Distributed_Extrinsics_Loss(coes, global_extrinsics, latent_extrinsics)));
      }
    };


    template<typename T>
    struct Distributed_Intrinsics_Loss{
      const Intrinsic_Coes* coes;
      const T* global_intrinsics;
      const T* latent_intrinsics;

      explicit Distributed_Intrinsics_Loss(const Intrinsic_Coes* coes, const T* global_intrinsics, const T* latent_intrinsics):
          coes(coes),global_intrinsics(global_intrinsics),latent_intrinsics(latent_intrinsics){};

      bool operator()(
          const T*const intrinsics,
          T* out_residuals
      )const{
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(out_residuals) << (intrinsics[OFFSET_FOCAL_LENGTH] - global_intrinsics[OFFSET_FOCAL_LENGTH] + latent_intrinsics[OFFSET_FOCAL_LENGTH]) * sqrt(coes->rho_f),
           (intrinsics[OFFSET_PRINCIPAL_POINT_X] - global_intrinsics[OFFSET_PRINCIPAL_POINT_X] + latent_intrinsics[OFFSET_PRINCIPAL_POINT_X]) * sqrt(coes->rho_uv),
           (intrinsics[OFFSET_PRINCIPAL_POINT_Y] - global_intrinsics[OFFSET_PRINCIPAL_POINT_Y] + latent_intrinsics[OFFSET_PRINCIPAL_POINT_Y]) * sqrt(coes->rho_uv),
           (intrinsics[OFFSET_DISTO_K1] - global_intrinsics[OFFSET_DISTO_K1] + latent_intrinsics[OFFSET_DISTO_K1]) * sqrt(coes->rho_d),
           (intrinsics[OFFSET_DISTO_K2] - global_intrinsics[OFFSET_DISTO_K2] + latent_intrinsics[OFFSET_DISTO_K2]) * sqrt(coes->rho_d),
           (intrinsics[OFFSET_DISTO_K3] - global_intrinsics[OFFSET_DISTO_K3] + latent_intrinsics[OFFSET_DISTO_K3]) * sqrt(coes->rho_d);
        return true;
      }

      static int num_residuals(){ return 6; }

      static ceres::CostFunction *Create
          (
              const Intrinsic_Coes* coes,
              const T* global_intrinsics,
              const T* latent_intrinsics
          ){
        return
            (new ceres::AutoDiffCostFunction
                <Distributed_Intrinsics_Loss,6,6>(
                new Distributed_Intrinsics_Loss(coes, global_intrinsics, latent_intrinsics)));
      }
    };


  } // namespace sfm
} // namespace openMVG

#endif // DISTRIBUTED_BA_CERES_RESIDUAL_FUNCTOR
