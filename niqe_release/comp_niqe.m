function result = comp_niqe(base_dir)
    niqe_obj = calc_udata(base_dir);
    total_noex = [niqe_obj.npe, niqe_obj.mef, niqe_obj.vv, niqe_obj.dicm, niqe_obj.lime];
    result = mean(total_noex);
end