#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/types.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include <memory>
#include <string>
#include <fstream>
#include <openMVG/sfm/sfm.hpp>

#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/types.hpp"

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::sfm;
using namespace std;

int main(int argc, char **argv) {

    std::string sDataDir;
    std::string sOutDir;

    SfM_Data sfm_data;

    CmdLine cmd;
    cmd.add( make_option('i', sDataDir, "sfm_data_dir") );
    cmd.add( make_option('o', sOutDir, "out_dir") );

    try {
        if (argc == 1) throw std::string("Invalid command line parameter.");
        cmd.process(argc, argv);
    } catch (const std::string& s) {
        return EXIT_FAILURE;
    }

    std::ofstream os;
    os.open(stlplus::create_filespec(sOutDir, "graph.txt").c_str());

    if (!Load(sfm_data, stlplus::create_filespec(sDataDir, "sfm_data.bin"), ESfM_Data(ALL))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \"" << "\" cannot be read." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << sfm_data.views.size() << " " << sfm_data.views[0]->id_pose << std::endl;
    uint32_t numViews = sfm_data.structure.size();
    vector<string> graph(numViews,"");
    int edgeCount = 0;
//    std::size_t numX3d = sfm_data.structure.size();
    for (auto & structure_landmark_it : sfm_data.structure){
        auto const observations = structure_landmark_it.second.obs;
        edgeCount += observations.size();
        for(auto & obs_it : observations){
            auto view_id = obs_it.first;
            auto pose_id = sfm_data.views[view_id]->id_pose;
            if(view_id != pose_id){
                std::cout << "view pose id mismatch:" << view_id << " " << pose_id << std::endl;
            }
            graph[view_id + 1] += std::to_string(structure_landmark_it.first + numViews + 1);
            graph[view_id + 1] += " ";
        }
    }
    graph[0] = std::to_string(numViews + sfm_data.structure.size()) + " " + std::to_string(edgeCount);
    for(uint32_t i=0; i<numViews;i++){
        os << graph[i] << std::endl;
    }
//    for(uint32_t i=0; i<sfm_data.structure.size(); i++){
//        os << std::endl;
//    }
    os.close();


    return EXIT_SUCCESS;
}
