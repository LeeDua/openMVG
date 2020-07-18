#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Cameras_Common_command_line_helper.hpp"
#include "software/distributed_ba/BA/distributed_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_report.hpp"
#include "openMVG/system/timer.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <cstdlib>
#include <memory>
#include <string>

using namespace openMVG;
using namespace openMVG::sfm;

int main(int argc, char **argv)
{
  using namespace std;

  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  std::string sIntrinsic_refinement_options = "NONE";
  bool b_use_motion_priors = false;

  cmd.add( make_option('i', sSfM_Data_Filename, "pre BA binary sfm data file") );
  cmd.add( make_option('o', sOutDir, "outdir") );
  cmd.add( make_option('f', sIntrinsic_refinement_options, "refineIntrinsics") );

  try {
    if (argc == 1) throw std::string("Invalid parameter.");
    cmd.process(argc, argv);
  } catch (const std::string& s) {
    std::cerr << "Usage: " << argv[0] << '\n'
              << "[-i|--input_file] path to a SfM_Data scene\n"
              << "[-m|--matchdir] path to the matches that corresponds to the provided SfM_Data scene\n"
              << "[-o|--outdir] path where the output data will be stored\n"
              << "\n[Optional]\n"
              << "[-r|--rotationAveraging]\n"
              << "\t 1 -> L1 minimization\n"
              << "\t 2 -> L2 minimization (default)\n"
              << "[-t|--translationAveraging]:\n"
              << "\t 1 -> L1 minimization\n"
              << "\t 2 -> L2 minimization of sum of squared Chordal distances\n"
              << "\t 3 -> SoftL1 minimization (default)\n"
              << "[-f|--refineIntrinsics] Intrinsic parameters refinement option\n"
              << "\t ADJUST_ALL -> refine all existing parameters (default) \n"
              << "\t NONE -> intrinsic parameters are held as constant\n"
              << "\t ADJUST_FOCAL_LENGTH -> refine only the focal length\n"
              << "\t ADJUST_PRINCIPAL_POINT -> refine only the principal point position\n"
              << "\t ADJUST_DISTORTION -> refine only the distortion coefficient(s) (if any)\n"
              << "\t -> NOTE: options can be combined thanks to '|'\n"
              << "\t ADJUST_FOCAL_LENGTH|ADJUST_PRINCIPAL_POINT\n"
              <<      "\t\t-> refine the focal length & the principal point position\n"
              << "\t ADJUST_FOCAL_LENGTH|ADJUST_DISTORTION\n"
              <<      "\t\t-> refine the focal length & the distortion coefficient(s) (if any)\n"
              << "\t ADJUST_PRINCIPAL_POINT|ADJUST_DISTORTION\n"
              <<      "\t\t-> refine the principal point position & the distortion coefficient(s) (if any)\n"
              << "[-P|--prior_usage] Enable usage of motion priors (i.e GPS positions)\n"
              << "[-M|--match_file] path to the match file to use.\n"
              << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }

  const cameras::Intrinsic_Parameter_Type intrinsic_refinement_options =
      cameras::StringTo_Intrinsic_Parameter_Type(sIntrinsic_refinement_options);
  if (intrinsic_refinement_options == static_cast<cameras::Intrinsic_Parameter_Type>(0) )
  {
    std::cerr << "Invalid input for Bundle Adjusment Intrinsic parameter refinement option" << std::endl;
    return EXIT_FAILURE;
  }

  // Load input SfM_Data scene
  SfM_Data scene_data;
  SfM_Data global_data;

  if (!Load(scene_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS|EXTRINSICS|STRUCTURE))) {
    std::cerr << std::endl
              << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }
  if (!Load(global_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS|EXTRINSICS|STRUCTURE))) {
    std::cerr << std::endl
              << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  if (sOutDir.empty())  {
    std::cerr << "\nIt is an invalid output directory" << std::endl;
    return EXIT_FAILURE;
  }

  if (!stlplus::folder_exists(sOutDir))
  {
    if (!stlplus::folder_create(sOutDir))
    {
      std::cerr << "\nCannot create the output directory" << std::endl;
    }
  }

  openMVG::system::Timer timer;

  int total_obs = 0;
  for(auto& track_it: global_data.structure){
    total_obs += track_it.second.obs.size();
  }

  Global_Params global_params(
      global_data.intrinsics.size(),
      global_data.poses.size(),
      total_obs,
      global_data.structure.size()
      );
  double QN_ratio = double(global_params.total_observations) / global_params.total_poses;
  Intrinsic_Coes intrinsic_coes(
      1e-3*QN_ratio,
      1e-3*QN_ratio,
      1e-4*QN_ratio
      );
  Extrinsic_Coes extrinsic_coes(
      1e5*QN_ratio,
      1e5*QN_ratio
      );
  Distributed_Bundle_Adjustment_Ceres ba_obj(
      global_data,
      &intrinsic_coes,
      &extrinsic_coes
      );

  Optimize_Options optimize_opt(
      intrinsic_refinement_options,
      Extrinsic_Parameter_Type::ADJUST_ALL,
      Structure_Parameter_Type::ADJUST_ALL,
      Control_Point_Parameter(0.0, false),
      false
      );
  ba_obj.Adjust(
      scene_data,
      optimize_opt
      );




  return EXIT_FAILURE;
}
