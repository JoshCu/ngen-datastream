import geopandas as gpd
import pandas as pd
import argparse
import re, os
import pickle, copy
from pathlib import Path
import datetime
gpd.options.io_engine = "pyogrio"

from ngen.config_gen.file_writer import DefaultFileWriter
from ngen.config_gen.hook_providers import DefaultHookProvider
from ngen.config_gen.generate import generate_configs

from ngen.config_gen.models.cfe import Cfe
from ngen.config_gen.models.pet import Pet

from ngen.config.realization import NgenRealization
from ngen.config.configurations import Routing

def gen_noah_owp_confs_from_pkl(pkl_file,out_dir,start,end):

    if not os.path.exists(out_dir):
        os.system(f"mkdir -p {out_dir}")

    with open(pkl_file, 'rb') as fp:
        nom_dict = pickle.load(fp)

    for jcatch in nom_dict:
        jcatch_str = copy.deepcopy(nom_dict[jcatch])
        for j,jline in enumerate(jcatch_str):
            if "startdate" in jline:
                pattern = r'(startdate\s*=\s*")[0-9]{12}'
                jcatch_str[j] = re.sub(pattern, f"startdate        = \"{start.strftime('%Y%m%d%H%M')}", jline)
            if "enddate" in jline:
                pattern = r'(enddate\s*=\s*")[0-9]{12}'
                jcatch_str[j] =  re.sub(pattern, f"enddate          = \"{end.strftime('%Y%m%d%H%M')}", jline)

        with open(Path(out_dir,f"noah-owp-modular-init-{jcatch}.namelist.input"),"w") as fp:
            fp.writelines(jcatch_str)

def generate_troute_conf(out_dir,start,max_loop_size):

    template = Path(__file__).parent.parent.parent.parent/"configs/ngen/ngen.yaml"

    with open(template,'r') as fp:
        conf_template = fp.readlines()

    for j,jline in enumerate(conf_template):
        if "qts_subdivisions" in jline:
            qts_subdivisions = int(jline.strip().split(': ')[-1])

    nts = max_loop_size * qts_subdivisions
      
    troute_conf_str = conf_template
    for j,jline in enumerate(conf_template):
        if "start_datetime" in jline:
            pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
            troute_conf_str[j] = re.sub(pattern, start.strftime('%Y-%m-%d %H:%M:%S'), jline)   

        pattern = r'^\s*max_loop_size\s*:\s*\d+\.\d+'
        if re.search(pattern,jline):
            troute_conf_str[j] = re.sub(pattern,  f"    max_loop_size: {max_loop_size}      ", jline)                    

        pattern = r'^\s*nts\s*:\s*\d+\.\d+'
        if re.search(pattern,jline):
            troute_conf_str[j] = re.sub(pattern,  f"    nts: {nts}      ", jline)

    with open(Path(out_dir,"ngen.yaml"),'w') as fp:
        fp.writelines(troute_conf_str)  

def gen_petAORcfe(hf_file,out,models,include):
    for j, jmodel in enumerate(include):
        hf: gpd.GeoDataFrame = gpd.read_file(hf_file, layer="divides")
        hf_lnk_data: pd.DataFrame = gpd.read_file(hf_file,layer="model-attributes")
        hook_provider = DefaultHookProvider(hf=hf, hf_lnk_data=hf_lnk_data)
        jmodel_out = Path(out,'cat_config',jmodel)
        os.system(f"mkdir -p {jmodel_out}")
        file_writer = DefaultFileWriter(jmodel_out)
        generate_configs(
            hook_providers=hook_provider,
            hook_objects=[models[j]],
            file_writer=file_writer,
        )

# Austin's multiprocess example from chat 3/25
# import concurrent.futures as cf
# from functools import partial
# def generate_configs_multiprocessing(
#     hook_providers: Iterable["HookProvider"],
#     hook_objects: Collection[BuilderVisitableFn],
#     file_writer: FileWriter,
#     pool: cf.ProcessPoolExecutor,
# ):
#     def capture(divide_id: str, bv: BuilderVisitableFn):
#         bld_vbl = bv()
#         bld_vbl.visit(hook_prov)
#         model = bld_vbl.build()
#         file_writer(divide_id, model)

#     div_hook_obj = DivideIdHookObject()
#     for hook_prov in hook_providers:
#         # retrieve current divide id
#         div_hook_obj.visit(hook_prov)
#         divide_id = div_hook_obj.divide_id()
#         assert divide_id is not None

#         fn = partial(capture, divide_id=divide_id)
#         pool.map(fn, hook_objects)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_file",
        dest="hf_file", 
        type=str,
        help="Path to the .gpkg", 
        required=False
    )
    parser.add_argument(
        "--outdir",
        dest="outdir", 
        type=str,
        help="Path to write ngen configs", 
        required=False
    )    
    parser.add_argument(
        "--pkl_file",
        dest="pkl_file", 
        type=str,
        help="Path to the noahowp pkl", 
        required=False
    )      
    parser.add_argument(
        "--realization",
        dest="realization", 
        type=str,
        help="Path to the ngen realization", 
        required=False
    )     
    parser.add_argument(
        "--ignore",
        dest="ignore", 
        type=str,
        help="List of NextGen BMI modules to ignore config generation for", 
        required=False
    )    

    args = parser.parse_args()
    ignore = args.ignore.split(',')

    serialized_realization = NgenRealization.parse_file(args.realization)
    start = serialized_realization.time.start_time
    end   = serialized_realization.time.end_time    
    max_loop_size = (end - start + datetime.timedelta(hours=1)).total_seconds() / (serialized_realization.time.output_interval)
    models = []
    include = []
    ii_cfe_or_pet = False
    model_names = []
    for jform in serialized_realization.global_config.formulations:
        for jmod in jform.params.modules:
            model_names.append(jmod.params.model_name)

    if "PET" in model_names:
        models.append(Pet)
        include.append("PET")
        ii_cfe_or_pet = True            
    if "CFE" in model_names:
        models.append(Cfe)    
        include.append("CFE")
        ii_cfe_or_pet = True            

    if "NoahOWP" in model_names:
        if "NoahOWP" in ignore:
            print(f'ignoring NoahOWP')
        else:
            if "pkl_file" in args:
                print(f'Generating NoahOWP configs from pickle',flush = True)
                noah_dir = Path(args.outdir,'cat_config','NOAH-OWP-M')
                os.system(f'mkdir -p {noah_dir}')
                gen_noah_owp_confs_from_pkl(args.pkl_file, noah_dir, start, end)
            else:
                raise Exception(f"Generating NoahOWP configs manually not implemented, create pkl.")            

    if ii_cfe_or_pet: 
        if "CFE" in ignore or "PET" in ignore:
            print(f'ignoring CFE and PET')
        else:
            print(f'Generating {include} configs from pydantic models',flush = True)
            gen_petAORcfe(args.hf_file,args.outdir,models,include)

    globals = [x[0] for x in serialized_realization]
    if serialized_realization.routing is not None:
        if "routing" in ignore:
            print(f'ignoring routing')
        else:
            print(f'Generating t-route config from template',flush = True)
            generate_troute_conf(args.outdir,start,max_loop_size) 

    print(f'Done!',flush = True)