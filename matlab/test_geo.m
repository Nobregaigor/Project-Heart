% nodes_file = "C:/Users/igorp/University of South Florida/Wenbin Mao - Igor/Febio-Models/Active-Models/LV/geometry_data/LV_myo_hexbase_nodes.csv";
% elem_file = "C:/Users/igorp/University of South Florida/Wenbin Mao - Igor/Febio-Models/Active-Models/LV/geometry_data/LV_myo_hexbase_elems.csv";

nodes_file = "./nodes.csv";
elem_file = "./elems.csv";

nodes = csvread(nodes_file, 0, 0);
elements = csvread(elem_file, 0, 0);

[model, mesh] = create_model(nodes', elements', true)



function [model, mesh] = create_model(nodes, elements, view_geometry)
    % fprintf('Creating Model...\n')
    model = createpde();
    [G, mesh] = geometryFromMesh(model, nodes, elements);

    if view_geometry
        %View the geometry and face numbers.
        pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5)
    end

    % fprintf('Model created.\n\n')
end