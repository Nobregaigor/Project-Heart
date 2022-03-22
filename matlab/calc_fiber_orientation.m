function [f0, s0] = calc_fiber_orientation( ...
        REF_TET_MESH, ... % struct containg nodes and elements of reference tet4 mesh
        MODEL_MESH, ... % struct containg nodes and elements of 'final' mesh
        APEX, ... % VIRTUAL NODE OF APEX
        BASE, ... % VIRTUAL NODE OF BASE
        THETA_ENDO, ...
        THETA_EPI)

    warning('off', 'all')
    %% Define Macros

    k = BASE - APEX;

    %% Load Model
    % create mesh, meshes must be triangular and tetrahedral elements
    [model, ~] = create_model(REF_TET_MESH.NODES', REF_TET_MESH.ELEMENTS', true);

    % %% Apply boundary conditions
    % % baseFace = 3;
    % endoFace = 1;
    % epicFace = 2;
    % apply_bcs(model, endoFace, epicFace); %base, endocardio, epicardio

    % % Create the mesh with target maximum element size 5: 'Hmax',5
    % generateMesh(model);

    % % Call the appropriate solver
    % result = solve_model(model, false);

    % %% interpolate back to the original meshes and nodes
    % elem_c = calculate_centers(MODEL_MESH.NODES, MODEL_MESH.ELEMENTS);

    % [u_interp, u_gradient] = interp_data(elem_c, result);

    % %% Sheet field direction

    % % Normal direction of the sheet based on interpolated gradient
    % s0 = sheet_normal_direction(u_gradient);

    % % Fiber angles at the endocardium and epicardium
    % f0 = fiber_direction(s0, k, u_interp, THETA_ENDO, THETA_EPI);

    %% Functions

    function [model, mesh] = create_model(nodes, elements, view_geometry)
        % fprintf('Creating Model...\n')
        model = createpde();
        [~, mesh] = geometryFromMesh(model, nodes, elements);

        if view_geometry
            %View the geometry and face numbers.
            pdegplot(model, 'FaceLabels', 'on', 'FaceAlpha', 0.5)
        end

        % fprintf('Model created.\n\n')
    end

    function apply_bcs(model, f1, f2)
        % fprintf('Applying BCs...\n')
        %Neumann bc on the base
        % applyBoundaryCondition(model, 'neumann', 'Face', base, 'q', 0, 'g', 0);
        %Create the boundary conditions
        applyBoundaryCondition(model, 'dirichlet', 'Face', f1, 'u', 0); %endocardium, f1
        applyBoundaryCondition(model, 'dirichlet', 'Face', f2, 'u', 1); %epicardium, f2
        %Create the PDE coefficients.
        specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', 0, 'f', 0);

        % fprintf('BCs applied\n\n')
    end

    function [result] = solve_model(model, plotSol)
        % fprintf('Solving model...\n')
        result = solvepde(model);

        %plot the solution
        if plotSol
            u = result.NodalSolution;
            figure
            pdeplot3D(model, 'ColorMapData', u)
        end

        % fprintf('Model solved\n\n')
    end

    function [elem_c] = calculate_centers(nodes_hex, elems_hex)
        % fprintf('Calculating centers...\n')
        nodes_hex = nodes_hex';
        elems_hex = elems_hex';

        L = length(elems_hex);
        elem_c = zeros(L, 3);

        for i = 1:L
            non_zero_elems = [];
            elems_line = (elems_hex(i, :));
            for j = 1:length(elems_line), if elems_line(j) > 0, non_zero_elems(j) = elems_line(j); end; end
            elem_c(i, :) = mean(nodes_hex(non_zero_elems, :), 1);
        end

    end

    function [u_interp, u_gradient] = interp_data(elem_c, model)
        % fprintf('Interpolating data...\n')
        u_interp = interpolateSolution(model, elem_c(:, 1), elem_c(:, 2), elem_c(:, 3));
        [gradx, grady, gradz] = evaluateGradient(model, elem_c(:, 1), elem_c(:, 2), elem_c(:, 3));
        u_gradient = [gradx, grady, gradz];
    end

    function [s0] = sheet_normal_direction(u_gradient)
        % fprintf('Calculating s0...\n')
        s0 = u_gradient ./ sqrt(u_gradient(:, 1).^2 +u_gradient(:, 2).^2 + u_gradient(:, 3).^2);
        % fprintf('s0 calculated.\n\n')
    end

    function [f0] = fiber_direction(s0, k, u_interp, theta_endo, theta_epi)
        % fprintf('Calculating fibers directions...\n')
        L = length(s0);
        kp = zeros(L, 3);

        for i = 1:length(s0)
            kp(i, :) = k - dot(k, s0(i, :)) * s0(i, :);
        end

        kp = kp ./ sqrt(kp(:, 1).^2 + kp(:, 2).^2 + kp(:, 3).^2);

        f0_t = cross(s0, kp, 2);
        theta = (theta_epi - theta_endo) * u_interp + theta_endo;
        f0 = [];

        for i = 1:length(s0)
            s0_cross = [0, -s0(i, 3), s0(i, 2); s0(i, 3), 0, -s0(i, 1); -s0(i, 2), s0(i, 1), 0];
            rot = eye(3) + sin(theta(i) * pi / 180) * s0_cross + 2 * sin(theta(i) * pi / 180/2)^2 * (s0(i, :)' * s0(i, :) - eye(3));
            f0(i, :) = rot * f0_t(i, :)';
        end

        % fprintf('Fibers directions calculated.\n\n')
    end

end
