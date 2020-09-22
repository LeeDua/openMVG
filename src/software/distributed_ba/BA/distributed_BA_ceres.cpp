// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "software/distributed_ba/BA/distributed_BA_ceres.hpp"

#ifdef OPENMVG_USE_OPENMP

#include <omp.h>

#endif

#include <mpi.h>
#include "openMVG/system/timer.hpp"

#include "ceres/problem.h"
#include "ceres/solver.h"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
//- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres_camera_functor.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/types.hpp"

#include <ceres/rotation.h>
#include <ceres/types.h>
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

using namespace std;

namespace openMVG {
    namespace sfm {

        using namespace openMVG::cameras;
        using namespace openMVG::geometry;


        class ExSyncCallback : public ceres::IterationCallback {
        public:
            explicit ExSyncCallback(map<uint32_t,MPI_Comm>&com_map, Hash_Map<IndexT, std::vector<double>>& local_ex_map, Hash_Map<IndexT, std::vector<double>>& global_ex_map, Hash_Map<IndexT, std::vector<double>>& latent_ex_map):
                com_map(com_map), local_ex_map(local_ex_map), global_ex_map(global_ex_map), latent_ex_map(latent_ex_map) {
            }

            ~ExSyncCallback() {}

            ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {

                MPI_Barrier(MPI_COMM_WORLD);
                int world_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                std::cout << "~~~~~~Custome iteration callback from ExSyncCallback~~~~~~~ PROC " << world_rank << std::endl;

                for (auto &com_it: com_map) {
                    int ex_id = com_it.first;
                    if(local_ex_map.count(ex_id)==0){
                        continue;
                    }

                    int com_size;
                    if (com_it.second == MPI_COMM_NULL) {
                        std::cout << "ERROR: NULL COMM" << std::endl;
//          std::cout << "Sub partition " << world_rank << " do not have cam" << com_it.first << std::endl;
                    } else {
                        MPI_Comm_size(com_it.second, &com_size);
                        if (com_size == 1) {
                            std::cout << "Cam " << com_it.first << " (single) of partition " << world_rank << std::endl;
                            for(int j=0; j<6; j++){
                                global_ex_map[ex_id][j] = local_ex_map[ex_id][j];
                                latent_ex_map[ex_id][j] = 0;
                            }
                        } else {
                            for(int j=0; j<6; j++){
                                global_ex_map[ex_id][j] = 0;
                            }
                            MPI_Request* request = new MPI_Request();
                            MPI_Iallreduce(&local_ex_map[ex_id][0], &global_ex_map[ex_id][0], 6, MPI_DOUBLE, MPI_SUM,
                                          com_it.second,request);
                            MPI_Wait(request,MPI_STATUS_IGNORE);
                            for (int j = 0; j < 6; j++) {
                                global_ex_map[ex_id][j] /= com_size;
                                latent_ex_map[ex_id][j] = latent_ex_map[ex_id][j] + (local_ex_map[ex_id][j] - global_ex_map[ex_id][j]);
                            }
                            std::cout << "Cam " << com_it.first << " (" << com_size << ") of partition " << world_rank
                                      << " average to "
                                      << global_ex_map[ex_id][0] << "(r1) "
                                      << global_ex_map[ex_id][1] << "(r2) "
                                      << global_ex_map[ex_id][2] << "(r3) "
                                      << global_ex_map[ex_id][3] << "(c1) "
                                      << global_ex_map[ex_id][4] << "(c2) "
                                      << global_ex_map[ex_id][0] << "(c3) "
                                      << std::endl;
                            MPI_Barrier(MPI_COMM_WORLD);
                        }
                    }
                }

                std::cout << "#####Custome iteration callback from ExSyncCallback END#####" << std::endl;
                return ceres::SOLVER_CONTINUE;
            }
        private:
            map<uint32_t,MPI_Comm>& com_map;
            Hash_Map<IndexT, std::vector<double>>& local_ex_map;
            Hash_Map<IndexT, std::vector<double>>& global_ex_map;
            Hash_Map<IndexT, std::vector<double>>& latent_ex_map;
        };


        Distributed_Bundle_Adjustment_Ceres::BA_Ceres_options::BA_Ceres_options
                (
                        const bool bVerbose,
                        bool bmultithreaded
                )
                : bVerbose_(bVerbose),
                  nb_threads_(1),
                  parameter_tolerance_(1e-8), //~= numeric_limits<float>::epsilon()
                  bUse_loss_function_(true) {
#ifdef OPENMVG_USE_OPENMP
            nb_threads_ = omp_get_max_threads();
#endif // OPENMVG_USE_OPENMP
            if (!bmultithreaded)
                nb_threads_ = 1;

            bCeres_summary_ = true;

            // If Sparse linear solver are available
            // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
            if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)) {
                sparse_linear_algebra_library_type_ = ceres::SUITE_SPARSE;
                linear_solver_type_ = ceres::ITERATIVE_SCHUR;
                preconditioner_type_ = ceres::CLUSTER_JACOBI;
//        preconditioner_type_ = ceres::CLUSTER_TRIDIAGONAL;
                visibility_clustering_type_ = ceres::SINGLE_LINKAGE;
//        visibility_clustering_type_ = ceres::CANONICAL_VIEWS;
            } else {
                std::cerr << "SUITSPARSE NOT SUPPORTED!" << std::endl;
                exit(-1);
            }
        }


        Distributed_Bundle_Adjustment_Ceres::Distributed_Bundle_Adjustment_Ceres
                (
                        Intrinsic_Coes *intrinsic_coes,
                        Extrinsic_Coes *extrinsic_coes,
                        const Distributed_Bundle_Adjustment_Ceres::BA_Ceres_options &options
                )
                : intrinsic_coes(intrinsic_coes), extrinsic_coes(extrinsic_coes), ceres_options_(options) {}

        Distributed_Bundle_Adjustment_Ceres::BA_Ceres_options &
        Distributed_Bundle_Adjustment_Ceres::ceres_options() {
            return ceres_options_;
        }

        bool Distributed_Bundle_Adjustment_Ceres::Adjust
                (
                        SfM_Data &sfm_data,     // the SfM scene to refine
                        const Optimize_Options &options
                ) {

//            MPI_Init(NULL, NULL);
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Group world_group;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);

            ifstream groupStream;
            map <uint32_t, vector<int>> groups;
            map <uint32_t, MPI_Comm> com_map;
            string groupDir("/home/spark/cloudleedua/output_data/");
            groupStream.open(stlplus::create_filespec(groupDir, "group.txt"));
            if (groupStream.fail()) {
                std::cerr << std::endl << "Input group txt file do not exist, id:" << world_rank << std::endl;
                groupStream.close();
                return EXIT_FAILURE;
            }
            std::cout << "Start reading sub partition" << world_rank << " communicator info" << std::endl;
            uint32_t pose_id;
            int comm_id;
            string line;
            while (std::getline(groupStream, line)) {
                std::stringstream ls(line);
                ls >> pose_id;
                while (ls >> comm_id) {
                    groups[pose_id].emplace_back(comm_id);
                }
            }
            std::cout << "Read sub partition " << world_rank << std::endl;
            for (auto &ex_it: groups) {
//                if(world_rank == 0){
//                    std::cout << "GROUP " << ex_it.first << " ";
//                    for(auto& c: ex_it.second){
//                        std::cout << c << " ";
//                    }
//                    std::cout << std::endl;
//                }
                MPI_Group g;
                MPI_Group_incl(world_group, ex_it.second.size(), &ex_it.second[0], &g);
                MPI_Comm comm = MPI_COMM_NULL;
                if(ex_it.second.size()>1){
                    MPI_Comm_create_group(MPI_COMM_WORLD, g, 0, &comm);
                    com_map[ex_it.first] = comm;
                }
            }
            std::cout << "Construct communicator map for sub partition " << world_rank << " , total communicators "
                      << com_map.size() << std::endl;

//            openMVG::system::Timer timer;
//            openMVG::system::Timer total;
//
//            for (auto &com_it: com_map) {
//                int ex_id = com_it.first;
//                int sum = 0;
//                int com_size;
//                if (com_it.second == MPI_COMM_NULL) {
//                    std::cout << "";
////          std::cout << "Sub partition " << world_rank << " do not have cam" << com_it.first << std::endl;
//                } else {
//                    MPI_Comm_size(com_it.second, &com_size);
//                    if (com_size == 1) {
//                        std::cout << "Cam " << com_it.first << " (single) of partition " << world_rank << std::endl;
//                    } else {
//                        timer.reset();
//                        MPI_Allreduce(&ex_id, &sum, 1, MPI_INT, MPI_SUM,
//                                      com_it.second);
//
//                        float average = sum * 1.0 / com_size;
//                        std::cout << "Cam " << com_it.first << " (" << com_size << ") of partition " << world_rank
//                                  << " average to " << average << std::endl;
//
//                        const Pose3 &pose = sfm_data.poses[ex_id];
//                        const Mat3 R = pose.rotation();
//                        const Vec3 c = pose.center();
//
//                        double angleAxis[3];
//                        ceres::RotationMatrixToAngleAxis((const double *) R.data(), angleAxis);
//
//                        double ex[6] = {angleAxis[0], angleAxis[1], angleAxis[2], c(0), c(1), c(2)};
//                        double sum[6] = {0, 0, 0, 0, 0, 0};
//                        MPI_Allreduce(ex, &sum, 6, MPI_DOUBLE, MPI_SUM,
//                                      com_it.second);
//                        for (int j = 0; j < 6; j++) {
//                            sum[j] /= com_size;
//                        }
//                        std::cout << "Cam " << com_it.first << " (" << com_size << ") of partition " << world_rank
//                                  << " average to "
//                                  << ex[0] << "(r1) "
//                                  << ex[1] << "(r2) "
//                                  << ex[2] << "(r3) "
//                                  << ex[3] << "(c1) "
//                                  << ex[4] << "(c2) "
//                                  << ex[0] << "(c3) "
//                                  << std::endl;
//                        std::cout << "Sync of cam " << ex_id << " on PROC " << world_rank << " took "
//                                  << timer.elapsedMs() << " ms" << std::endl;
//                    }
//
//                }
//            }
//            std::cout << "Total sync time for all cams (" << com_map.size() << ") take " << total.elapsedMs() << " ms"
//                      << std::endl;
//            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "BA adjust start" << std::endl;
            //----------
            // Add camera parameters
            // - intrinsics
            // - poses [R|t]

            // Create residuals for each observation in the bundle adjustment problem. The
            // parameters for cameras and points are added automatically.
            //----------

            bool b_usable_prior = false;
            if (options.use_motion_priors_opt) {
                std::cerr << "In distributed ba, use motion priors, not implemented" << std::endl;
                return false;
            }

            ceres::Problem problem;

            // extract global extrinsic data and intrinsic data
            Hash_Map <IndexT, std::vector<double>> global_intrinsics;
            Hash_Map <IndexT, std::vector<double>> latent_intrinsics;
            Hash_Map <IndexT, std::vector<double>> map_intrinsics;
            Hash_Map <IndexT, std::vector<double>> global_poses;
            Hash_Map <IndexT, std::vector<double>> latent_poses;
            Hash_Map <IndexT, std::vector<double>> map_poses;


            for (const auto &intrinsic_it : sfm_data.intrinsics) {
                const IndexT indexCam = intrinsic_it.first;
                global_intrinsics[indexCam] = intrinsic_it.second->getParams();
                std::vector<double> latent_init(global_intrinsics[indexCam].size(), 0.0);
                latent_intrinsics[indexCam] = std::move(latent_init);
            }

            // Setup Poses data & subparametrization
            for (const auto &pose_it : sfm_data.poses) {
                const IndexT indexPose = pose_it.first;

                const Pose3 &pose = pose_it.second;
                const Mat3 R = pose.rotation();
                const Vec3 c = pose.center();

                double angleAxis[3];
                ceres::RotationMatrixToAngleAxis((const double *) R.data(), angleAxis);
                // angleAxis + translation
                map_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], c(0), c(1), c(2)};
                global_poses[indexPose] = {angleAxis[0], angleAxis[1], angleAxis[2], c(0), c(1), c(2)};
                latent_poses[indexPose] = {0, 0, 0, 0, 0, 0};

                double *parameter_block = &map_poses.at(indexPose)[0];
                problem.AddParameterBlock(parameter_block, 6);
                if (options.extrinsics_opt != Extrinsic_Parameter_Type::ADJUST_ALL) {
                    std::cerr << "Should adjust all extrinsics in distributed BA" << std::endl;
                }
            }

            std::cout << "initialization done" << std::endl;


            // Setup Intrinsics data & subparametrization
            for (const auto &intrinsic_it : sfm_data.intrinsics) {
                const IndexT indexCam = intrinsic_it.first;

                if (isValid(intrinsic_it.second->getType())) {
                    map_intrinsics[indexCam] = intrinsic_it.second->getParams();
                    if (!map_intrinsics.at(indexCam).empty()) {
                        double *parameter_block = &map_intrinsics.at(indexCam)[0];
                        problem.AddParameterBlock(parameter_block, map_intrinsics.at(indexCam).size());
                        if (options.intrinsics_opt == Intrinsic_Parameter_Type::NONE) {
                            // set the whole parameter block as constant for best performance
                            problem.SetParameterBlockConstant(parameter_block);
                        } else {
                            const std::vector<int> vec_constant_intrinsic =
                                    intrinsic_it.second->subsetParameterization(options.intrinsics_opt);
                            if (!vec_constant_intrinsic.empty()) {
                                ceres::SubsetParameterization *subset_parameterization =
                                        new ceres::SubsetParameterization(
                                                map_intrinsics.at(indexCam).size(), vec_constant_intrinsic);
                                problem.SetParameterization(parameter_block, subset_parameterization);
                            }
                        }
                    }
                } else {
                    std::cerr << "Unsupported camera type." << std::endl;
                    return false;
                }
            }

            std::cout << "intrinsic parameterization done" << std::endl;


            // Set a LossFunction to be less penalized by false measurements
            //  - set it to nullptr if you don't want use a lossFunction.
            ceres::LossFunction *p_LossFunction = nullptr;

            // For all visibility add reprojections errors:
            for (auto &structure_landmark_it : sfm_data.structure) {
                const Observations &obs = structure_landmark_it.second.obs;

                for (const auto &obs_it : obs) {
                    // Build the residual block corresponding to the track observation:
                    const View *view = sfm_data.views.at(obs_it.first).get();

                    // Each Residual block takes a point and a camera as input and outputs a 2
                    // dimensional residual. Internally, the cost function stores the observed
                    // image location and compares the reprojection against the observation.
//                    ceres::CostFunction* cost_function =
//                            IntrinsicsToCostFunction(sfm_data.intrinsics.at(view->id_intrinsic).get(),
//                                                     obs_it.second.x);
                    if (sfm_data.intrinsics.at(view->id_intrinsic).get()->getType() != PINHOLE_CAMERA_RADIAL3) {
                        std::cerr << "Camera model other than PINHOLE_CAMERA_RADIAL3 is not implemented yet"
                                  << std::endl;
                        exit(-1);
                    }

                    //naive reprojection loss
                    ceres::CostFunction *cost_function =
                            Distributed_ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::Create(obs_it.second.x);

                    if (cost_function) {
                        if (!map_intrinsics.at(view->id_intrinsic).empty()) {
                            problem.AddResidualBlock(cost_function,
                                                     p_LossFunction,
                                                     &map_intrinsics.at(view->id_intrinsic)[0],
                                                     &map_poses.at(view->id_pose)[0],
                                                     structure_landmark_it.second.X.data());
                        } else {
                            std::cerr << "cost function set to nullptr, illegal" << std::endl;
                            return false;
//              problem.AddResidualBlock(cost_function,
//                                       p_LossFunction,
//                                       &map_poses.at(view->id_pose)[0],
//                                       structure_landmark_it.second.X.data());
                        }
                    } else {
                        std::cerr << "Cannot create a CostFunction for this camera model." << std::endl;
                        return false;
                    }
                }
                if (options.structure_opt == Structure_Parameter_Type::NONE) {
                    std::cerr << "structure_opt set to NONE, illegal" << std::endl;
                    return false;
                    //          problem.SetParameterBlockConstant(structure_landmark_it.second.X.data());
                }
            }

            std::cout << "reprojection loss added" << std::endl;


            if (false) {
                for (auto &pose_it : map_poses) {
                    IndexT pose_id = pose_it.first;
                    if (global_poses.find(pose_id) == global_poses.end()) {
                        std::cerr << "global_poses do not contain pose " << pose_id << std::endl;
                    }
                    auto &pose_global = global_poses[pose_id];
                    auto &pose_latent = latent_poses[pose_id];
                    ceres::CostFunction *extrinsic_cost = Distributed_Extrinsics_Loss::Create(extrinsic_coes,
                                                                                              &pose_global[0],
                                                                                              &pose_latent[0]);
                    problem.AddResidualBlock(extrinsic_cost,
                                             nullptr,
                                             &pose_it.second[0]);

                }
            }

            std::cout << "extrinsic loss configured" << std::endl;


            if (options.intrinsics_opt != Intrinsic_Parameter_Type::NONE) {
                for (auto &intrinsic_it : map_intrinsics) {
                    IndexT intrinsic_id = intrinsic_it.first;
                    if (global_intrinsics.find(intrinsic_id) == global_intrinsics.end()) {
                        std::cerr << "global intrinsics do not contain camera " << intrinsic_id << std::endl;
                    }
                    auto &intrinsic_global = global_intrinsics[intrinsic_id];
                    auto &intrinsic_latent = latent_intrinsics[intrinsic_id];
                    ceres::CostFunction *intrinsic_cost = Distributed_Intrinsics_Loss::Create(intrinsic_coes,
                                                                                              &intrinsic_global[0],
                                                                                              &intrinsic_latent[0]);
                    problem.AddResidualBlock(intrinsic_cost,
                                             nullptr,
                                             &intrinsic_it.second[0]);
                }
            }

            std::cout << "intrinsic loss configured" << std::endl;


            if (options.control_point_opt.bUse_control_points) {
                std::cerr << "In distributed BA, use control points, not implemented" << std::endl;
                return false;
            }

            // Add Pose prior constraints if any
            if (b_usable_prior) {
                std::cerr << "In distributed ba, use prior, not implemented" << std::endl;
                return false;
            }

            // Configure a BA engine and run it
            //  Make Ceres automatically detect the bundle structure.
            ceres::Solver::Options ceres_config_options;
            ceres_config_options.max_num_iterations = 500;
            ceres_config_options.preconditioner_type =
                    static_cast<ceres::PreconditionerType>(ceres_options_.preconditioner_type_);
            ceres_config_options.visibility_clustering_type =
                    static_cast<ceres::VisibilityClusteringType>(ceres_options_.visibility_clustering_type_);
            ceres_config_options.linear_solver_type =
                    static_cast<ceres::LinearSolverType>(ceres_options_.linear_solver_type_);
            ceres_config_options.sparse_linear_algebra_library_type =
                    static_cast<ceres::SparseLinearAlgebraLibraryType>(ceres_options_.sparse_linear_algebra_library_type_);
            ceres_config_options.minimizer_progress_to_stdout = ceres_options_.bVerbose_;
            ceres_config_options.logging_type = ceres::PER_MINIMIZER_ITERATION;
            ceres_config_options.num_threads = ceres_options_.nb_threads_;
#if CERES_VERSION_MAJOR < 2
            ceres_config_options.num_linear_solver_threads = ceres_options_.nb_threads_;
#endif
            ceres_config_options.parameter_tolerance = ceres_options_.parameter_tolerance_;
            ceres_config_options.update_state_every_iteration = true;
//
//            ceres_config_options.callbacks.push_back(new ExSyncCallback(
//                    com_map,map_poses,global_poses,latent_poses
//                    ));

            // Solve BA
            std::cout << "Before actual ba calculation" << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
            ceres::Solver::Summary summary;
            ceres::Solve(ceres_config_options, &problem, &summary);
            std::cout << "End ba calculation" << std::endl;

            if (ceres_options_.bCeres_summary_)
                std::cout << summary.FullReport() << std::endl;

            // If no error, get back refined parameters
            if (!summary.IsSolutionUsable()) {
                if (ceres_options_.bVerbose_)
                    std::cout << "Bundle Adjustment failed." << std::endl;
                return false;
            } else // Solution is usable
            {
                if (ceres_options_.bVerbose_) {
                    // Display statistics about the minimization
                    std::cout << std::endl
                              << "Bundle Adjustment statistics (approximated RMSE):\n"
                              << " #views: " << sfm_data.views.size() << "\n"
                              << " #poses: " << sfm_data.poses.size() << "\n"
                              << " #intrinsics: " << sfm_data.intrinsics.size() << "\n"
                              << " #tracks: " << sfm_data.structure.size() << "\n"
                              << " #residuals: " << summary.num_residuals << "\n"
                              << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
                              << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
                              << " Time (s): " << summary.total_time_in_seconds << "\n"
                              << std::endl;
                    if (options.use_motion_priors_opt)
                        std::cout << "Usable motion priors: " << (int) b_usable_prior << std::endl;
                }

                // Update camera poses with refined data
                if (options.extrinsics_opt != Extrinsic_Parameter_Type::NONE) {
                    for (auto &pose_it : sfm_data.poses) {
                        const IndexT indexPose = pose_it.first;

                        Mat3 R_refined;
                        ceres::AngleAxisToRotationMatrix(&map_poses.at(indexPose)[0], R_refined.data());
                        Vec3 C_refined(map_poses.at(indexPose)[3], map_poses.at(indexPose)[4],
                                       map_poses.at(indexPose)[5]);
                        // Update the pose
                        Pose3 &pose = pose_it.second;
                        pose = Pose3(R_refined, C_refined);
                    }
                }

                // Update camera intrinsics with refined data
                if (options.intrinsics_opt != Intrinsic_Parameter_Type::NONE) {
                    for (auto &intrinsic_it : sfm_data.intrinsics) {
                        const IndexT indexCam = intrinsic_it.first;

                        const std::vector<double> &vec_params = map_intrinsics.at(indexCam);
                        intrinsic_it.second->updateFromParams(vec_params);
                    }
                }

                // Structure is already updated directly if needed (no data wrapping)

                if (b_usable_prior) {
                    std::cerr << "In distributed ba, use prior, not implemented" << std::endl;
                    return false;
                }
                return true;
            }
        }

    } // namespace sfm
} // namespace openMVG
