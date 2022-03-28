clc; clear;
% nodes_file = "C:/Users/igorp/University of South Florida/Wenbin Mao - Igor/Febio-Models/Active-Models/LV/geometry_data/LV_myo_hexbase_nodes.csv";
% elem_file = "C:/Users/igorp/University of South Florida/Wenbin Mao - Igor/Febio-Models/Active-Models/LV/geometry_data/LV_myo_hexbase_elems.csv";

nodes_file = "./nodes.csv";
elem_file = "./elems.csv";
regionIds_file = "./regionIDs.csv";

nodes = csvread(nodes_file, 0, 0);
elements = csvread(elem_file, 0, 0);
regionIds = csvread(regionIds_file, 0, 0);

model = createpde();
[G, mesh] = geometryFromMesh(model, nodes', elements', regionIds);
pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5)