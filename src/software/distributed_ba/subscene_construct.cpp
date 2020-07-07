#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/types.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/cmdLine/cmdLine.h"
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
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
    std::string sGraphDir;
    std::string sLandmarkMapDir;
    std::string sOutDir;
    int K = 0;

    SfM_Data sfm_data;

    CmdLine cmd;
    cmd.add( make_option('i', sDataDir, "sfm_data_dir") );
    cmd.add( make_option('g', sGraphDir, "graph_file") );
    cmd.add( make_option('m', sLandmarkMapDir, "landmark_map_file") );
    cmd.add( make_option('k', K, "partitions") );
    cmd.add( make_option('o', sOutDir, "out_dir") );

    try {
        if (argc == 1) throw std::string("Invalid command line parameter.");
        cmd.process(argc, argv);
    } catch (const std::string& s) {
        return EXIT_FAILURE;
    }

    if ( K <= 0 ){
        std::cerr << "Should specify partition count with value > 0, current K:" << K << std::endl;
    }

    if (!Load(sfm_data, stlplus::create_filespec(sDataDir, "sfm_data.bin"), ESfM_Data(ALL))) {
        std::cerr << std::endl
                  << "The input SfM_Data file \"" << "\" cannot be read." << std::endl;
        return EXIT_FAILURE;
    }

    std::map<uint32_t,uint32_t> trackMap;
    ifstream mapStream;
    mapStream.open(sLandmarkMapDir);
    if(mapStream.fail()){
        std::cerr << std::endl << "Input landmark map file do not exist" << std::endl;
        mapStream.close();
        return EXIT_FAILURE;
    }
    ifstream graphStream;
    graphStream.open(sGraphDir);
    if(mapStream.fail()){
        std::cerr << std::endl << "Input graph file do not exist" << std::endl;
        graphStream.close();
        return EXIT_FAILURE;
    }

    uint32_t from, to;
    while(mapStream >> from >> to){
        trackMap.emplace(std::make_pair(from, to));
    }
    mapStream.close();
    uint32_t numTracks = sfm_data.structure.size();
    assert(trackMap.size() == numTracks);


    const uint32_t numViews = sfm_data.views.size();

    uint32_t vCount = 0;
    int partition = 0;
    vector<SfM_Data> subScenes(K);
    //copy root path
    for(auto& d : subScenes){
        d.s_root_path = sfm_data.s_root_path;
    }
    while(graphStream >> partition){
        if(vCount < numViews){
            vCount ++;
            //copy views during landmark copying
            continue;
        }
        uint32_t trackId = trackMap[vCount+1];
        Landmark landmark = sfm_data.structure[trackId];
        subScenes[partition].structure.emplace(std::make_pair(trackId, landmark));
        for(auto & obs_it : landmark.obs){
            uint32_t view_id = obs_it.first;
            Views& views = subScenes[partition].views;
            if(views.find(view_id) == views.end()){
                //copy view
                views.emplace(std::make_pair(view_id,sfm_data.views[view_id]));
                //copy pose
                subScenes[partition].poses.emplace(std::make_pair(view_id, sfm_data.poses[view_id]));
                //copy intrinsic
                uint32_t intrinsic_id = sfm_data.views[view_id]->id_intrinsic;
                Intrinsics& intrinsics =  subScenes[partition].intrinsics;
                if(intrinsics.find(intrinsic_id) ==intrinsics.end()){
                    intrinsics.emplace(std::make_pair(intrinsic_id, sfm_data.intrinsics[intrinsic_id]));
                }
            }
        }
        vCount ++;
    }
    graphStream.close();
    assert(vCount == numViews + numTracks);

    std::string bin_path = stlplus::create_filespec(sOutDir,"binpartitions"+std::to_string(K));
    std::string ply_path = stlplus::create_filespec(sOutDir,"plypartitions"+std::to_string(K));

    if(!stlplus::folder_exists(bin_path)){
        stlplus::folder_create(bin_path);
    }
    if(!stlplus::folder_exists(ply_path)){
        stlplus::folder_create(ply_path);
    }
    std::ofstream logStream(stlplus::create_filespec(bin_path, "stats.log"));
    for(uint32_t i=0;i<subScenes.size();i++){
        if(!Save(subScenes[i], stlplus::create_filespec(bin_path, "scene" + std::to_string(i) +".bin"),ESfM_Data::ALL)){
            std::cerr << "Save subscene " << i << " bin failed" << std::endl;
        };
        if(!Save(subScenes[i], stlplus::create_filespec(ply_path, "scene" + std::to_string(i) +".ply"),ESfM_Data::STRUCTURE)){
            std::cerr << "Save subscene " << i << " structure ply failed" << std::endl;
        };
        if(!Save(subScenes[i], stlplus::create_filespec(ply_path, "cam" + std::to_string(i) +".ply"),ESfM_Data::EXTRINSICS)){
            std::cerr << "Save subscene " << i << " camera ply failed" << std::endl;
        };
        int obCount = 0;
        for(auto& landmark_iter: subScenes[i].structure){
            obCount += landmark_iter.second.obs.size();
        }
        logStream << "SCENE" << i << std::endl;
        logStream << "rootPath " << " " <<  subScenes[i].s_root_path << std::endl;
        logStream << "numViews " << " " << subScenes[i].views.size() << std::endl;
        logStream << "numPoses " << " " << subScenes[i].poses.size() << std::endl;
        logStream << "numIntrs " << " " << subScenes[i].intrinsics.size() << std::endl;
        logStream << "numTracks" << " " << subScenes[i].structure.size() << std::endl;
        logStream << "numObsers" << " " << obCount << std::endl;
        logStream << std::endl;
    }
    logStream.close();

    return EXIT_SUCCESS;
}
