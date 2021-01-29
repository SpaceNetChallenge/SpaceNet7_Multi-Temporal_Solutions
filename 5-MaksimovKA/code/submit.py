import os
import tqdm
import glob
import fiona
import geopandas as gpd
from fire import Fire


def sn7_convert_geojsons_to_csv(json_dirs, output_csv_path, population='proposal'):
    '''
    Convert jsons to csv
    Population is either "ground" or "proposal"
    '''

    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm.tqdm(json_dirs[:]):
        json_files = sorted(glob.glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm.tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                # raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')

            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                # raise Exception(message)
                df = gpd.GeoDataFrame({
                    'filename': file_name_col,
                    'id': 0,
                    'geometry': "POLYGON EMPTY",
                })
            else:
                try:
                    df = gpd.GeoDataFrame({
                        'filename': file_name_col,
                        'id': df.Id.astype(int),
                        'geometry': df.geometry,
                    })
                except:
                    print(df)
            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)

    net_df.to_csv(output_csv_path, index=False)
    return net_df

def make_submit(out_file='/wdata/solution.csv'):
    pred_top_dir = '/wdata/'

    # out_dir_csv = os.path.join(pred_top_dir, 'csvs')
    # os.makedirs(out_dir_csv, exist_ok=True)
    # prop_file = os.path.join(out_dir_csv, 'solution.csv')
    prop_file = out_file
    aoi_dirs = sorted([os.path.join(pred_top_dir, 'pred_jsons_match', aoi) \
                       for aoi in os.listdir(os.path.join(pred_top_dir, 'pred_jsons_match')) \
                       if os.path.isdir(os.path.join(pred_top_dir, 'pred_jsons_match', aoi))])
    print("aoi_dirs:", aoi_dirs)

    # Execute
    if os.path.exists(prop_file):
        os.remove(prop_file)
    net_df = sn7_convert_geojsons_to_csv(aoi_dirs, prop_file, 'proposal')

    print("prop_file:", prop_file)

if __name__ == '__main__':
    Fire(make_submit)