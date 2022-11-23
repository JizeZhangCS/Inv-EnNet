classdef calc_udata
    properties
        dicm
        mef
        lime
        npe
        vv
    end
    methods 
        function obj = calc_udata(base_dir)
            obj.dicm = niqe_calc('DICM', base_dir);
            obj.mef = niqe_calc('MEF', base_dir);
            obj.lime = niqe_calc('LIME', base_dir);
            obj.npe = niqe_calc('NPE', base_dir);
            obj.vv = niqe_calc('VV', base_dir);
        end
        function disp(obj)
            disp('DICM, MEF, LIME, NPE, VV')
            [mean(obj.dicm), mean(obj.mef), mean(obj.lime), mean(obj.npe), mean(obj.vv)]
            disp('total_noex');
            total_noex = [obj.npe, obj.mef, obj.vv, obj.dicm, obj.lime];
            mean(total_noex)
        end
    end
end

function result = niqe_calc(dataset, base_dir)

    load modelparameters.mat
    blocksizerow    = 96;
    blocksizecol    = 96;
    blockrowoverlap = 0;
    blockcoloverlap = 0;
    
    curr_dir = strcat(base_dir, dataset, '/');
    list = dir(curr_dir);
    list = list(3:size(list));
    len = size(list);
    result = [];
    for idx = 1:size(list)
        img = imread(strcat(curr_dir, list(idx).name));
        result(end+1) = computequality(img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
        mu_prisparam,cov_prisparam);
    end
end