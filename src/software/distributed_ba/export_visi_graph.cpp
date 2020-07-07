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


    if (!Load(sfm_data, stlplus::create_filespec(sDataDir, "sfm_data.bin"), ESfM_Data(ALL))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \"" << "\" cannot be read." << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t numViews = sfm_data.views.size();
    vector<string> graph( 1 + numViews + sfm_data.structure.size(),"");
    int edgeCount = 0;
    uint32_t trackCount = 0;
    std::map<uint32_t,uint32_t> trackMap;

    std::cout << "total vs,views,x3ds " << graph.size() << " "  << numViews << " " << sfm_data.structure.size() << std::endl;

    std::ofstream os;
    os.open(stlplus::create_filespec(sOutDir, "graph.txt").c_str());
    std::ofstream trackStream;
    trackStream.open(stlplus::create_filespec(sOutDir, "track_map.txt").c_str());

    // set camera vertex weights to 0 and point vertex weight to 1
    // to avoid clustering only cameras without points
    for(int i=0;i<=numViews;i++){
        graph[i] += "0 ";
    }
    for(int i = numViews+1; i < graph.size(); i++){
        graph[i] += "1 ";
    }
    
    for (auto & structure_landmark_it : sfm_data.structure){
        auto const observations = structure_landmark_it.second.obs;
        trackMap.emplace(std::make_pair(trackCount + numViews + 1,structure_landmark_it.first));
        edgeCount += observations.size();
        for(auto & obs_it : observations){
            auto view_id = obs_it.first;
            auto pose_id = sfm_data.views[view_id]->id_pose;
            if(view_id != pose_id){
                std::cout << "view pose id mismatch:" << view_id << " " << pose_id << std::endl;
            }
            graph[view_id + 1] += std::to_string(trackCount + numViews + 1) + " ";
            graph[numViews + trackCount + 1] +=  std::to_string(view_id + 1) + " ";
        }
        trackCount += 1;
    }
    graph[0] = std::to_string(numViews + sfm_data.structure.size()) + " " + std::to_string(edgeCount) + " 010";
    for(uint32_t i=0; i< graph.size();i++){
        os << graph[i] << std::endl;
    }
//    for(uint32_t i=0; i<sfm_data.structure.size(); i++){
//        os << std::endl;
//    }
    for(auto iter=trackMap.cbegin();iter!=trackMap.cend();iter++){
        trackStream << iter->first << " " << iter->second << std::endl;
    }
    os.close();
    trackStream.close();


    return EXIT_SUCCESS;
}
