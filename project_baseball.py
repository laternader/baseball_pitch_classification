# Functions contained in here will help with the modeling process

def pitcher_stats(name ,df):
    '''
    name: name of pitcher as a string
    df: dataframe of the season you want to check
    '''
    try:
        return df[df['pitcher_name']==name]
    except:
        return 'Did not play this season or you misspelled his name.'

# Get Batter Stats
def batter_stats(name ,df):
    '''
    name: name of batter as a string
    df: dataframe of the season you want to check
    '''
    try:
        return df[df['batter_name']==name]
    except:
        return 'Did not play this season or you misspelled his name.'


    
### Cleaning Functions ###
def renaming_columns(mlb):
    mlb.rename(columns={'player_name':'pitcher_name'}, inplace=True) # Rename pitcher
    mlb.rename(columns={'fielder_2':'Catcher'}, inplace=True) # rename Catcher
    # Rename other fielders
    mlb.rename(columns={'fielder_3':'FirstBasemen',
                        'fielder_4':'SecondBasemen',
                        'fielder_5':'ThirdBasemen',
                        'fielder_6':'ShortStop',
                        'fielder_7':'LeftField',
                        'fielder_8':'CenterField',
                        'fielder_9':'RightField'}, inplace=True)
    
    return mlb

def dropping_columns(mlb):
    mlb.drop(columns=['spin_dir', 'spin_rate_deprecated',
       'break_angle_deprecated', 'break_length_deprecated','tfs_deprecated', 'tfs_zulu_deprecated',
        'umpire', 'sv_id', 'pitcher.1', 'fielder_2.1'], inplace=True)

    return mlb

### Feature Engineering Functions ###
def batter_name(des):
    try:
        name = ' '.join(des.split(' ',2)[:2])
        return name
    except:
        return np.nan

