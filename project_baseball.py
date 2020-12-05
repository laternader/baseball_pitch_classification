"""
Functions contained in here will help with the modeling process
"""
import numpy as np
import pandas as pd

# Global Variables
features = ['game_date','pitch_type','Pitcher_name', 'batter_id','pitcher_id','release_speed', 'release_pos_x', 'release_pos_z', 'stand', 'p_throws', 'balls', 'strikes',
            'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'inning', 'inning_topbot',
            'effective_speed', 'release_spin_rate', 'release_extension','Catcher','FirstBasemen', 'SecondBasemen', 'ThirdBasemen', 'ShortStop',
            'LeftField', 'CenterField', 'RightField', 'at_bat_number', 'pitch_number', 'pitch_name',
            'bat_score', 'fld_score', 'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment',
            'count', 'strike_attempt','events','description','type','bb_type','vx0','vy0','vz0',
            'ax','ay','az','outs_when_up']

ids = ['pitcher_id','batter_id','on_1b','on_2b','on_3b','Catcher','FirstBasemen',
      'SecondBasemen', 'ThirdBasemen', 'ShortStop', 'LeftField','CenterField', 'RightField']
make_binary = ['stand','p_throws']
integerify = ['outs_when_up','bat_score','fld_score','post_bat_score','post_fld_score','pitch_number',
             'at_bat_number','inning','balls','strikes']
f2 = ['game_date', 'pitch_type', 'Pitcher_name', 'pitcher_id', 'batter_name',
       'batter_id', 'release_speed', 'release_pos_x', 'release_pos_z', 'stand',
       'p_throws', 'balls', 'strikes', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
       'on_3b', 'on_2b', 'on_1b', 'inning', 'inning_topbot', 'effective_speed',
       'release_spin_rate', 'release_extension', 'Catcher', 'FirstBasemen',
       'SecondBasemen', 'ThirdBasemen', 'ShortStop', 'LeftField',
       'CenterField', 'RightField', 'at_bat_number', 'pitch_number',
       'pitch_name', 'bat_score', 'fld_score', 'post_bat_score',
       'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment',
       'count', 'strike_attempt', 'events', 'description', 'type', 'bb_type',
       'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'outs_when_up']

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
def renaming_fielders(mlb):
    mlb.rename(columns={'player_name':'Pitcher_name'}, inplace=True) # Rename pitcher
    mlb.rename(columns={'fielder_2':'Catcher'}, inplace=True) # rename Catcher
    # Rename other fielders
    mlb.rename(columns={'fielder_3':'FirstBasemen',
                        'fielder_4':'SecondBasemen',
                        'fielder_5':'ThirdBasemen',
                        'fielder_6':'ShortStop',
                        'fielder_7':'LeftField',
                        'fielder_8':'CenterField',
                        'fielder_9':'RightField'}, inplace=True)
    mlb.rename(columns={'batter':'batter_id',
                       'pitcher':'pitcher_id'}, inplace=True)
    return mlb

def dropping_columns(mlb):
    mlb.drop(columns=['spin_dir', 'spin_rate_deprecated',
       'break_angle_deprecated', 'break_length_deprecated','tfs_deprecated', 'tfs_zulu_deprecated',
       'umpire', 'sv_id', 'pitcher.1', 'fielder_2.1'], inplace=True)

    return mlb

### Feature Engineering Functions ###


### Cleaning Function Totality ###

def cleaning(df):
    df = renaming_fielders(df)
    df = dropping_columns(df)
    
    df['game_date']=pd.to_datetime(df['game_date'])
    df['pitch_type']=df['pitch_type'].map({'CS':'CU',
                           'FF':'FF',
                           'CU':'CU',
                           'CH':'CH',
                           'SI':'SI',
                           'SL':'SL',
                           'FC':'FC',
                           'KC':'KC',
                           'FS':'FS',
                           'KN':'KN',
                           'FO':'FO'})
#     df = df[df['release_speed'].notnull()]
    
    df['pitch_type'].fillna('U', inplace=True)
    df['pitch_name'].fillna('Unknown', inplace=True)
    df['if_fielding_alignment'].fillna('Unknown', inplace=True)
    df['of_fielding_alignment'].fillna('Unknown', inplace=True)
    
    return df

def strike_attempt_column(df):
    
    df['strike_attempt']=df['description'].map(
            {'called_strike':'strike',
             'swinging_strike':'strike',
             'ball':'ball',
             'foul':'strike', # False in out_via_description
             'hit_into_play':'out',
             'blocked_ball':'ball',
             'hit_into_play_score':'ob',
             'swinging_strike_blocked':'strike',
             'hit_into_play_no_out':'ob',
             'foul_bunt':'strike',
             'foul_tip':'strike',
             'hit_by_pitch':'ob',
             'missed_bunt':'strike',
             'pitchout':'out',
             'bunt_foul_tip':'strike',
             'intent_ball':'ob',
             'swinging_pitchout':'strike',
             'foul_pitchout':'strike',
             'pitchout_hit_into_play_score':'out'}).copy()
    
    return df

### These functions are a pair of mangoes ###
def batter_name(des):
    try:
        name = ' '.join(des.split(' ',2)[:2])
        return name
    except:
        return np.nan
    
def fill_in_batters(df):
    
    df['batter_name'] = df['des'].map(batter_name)
    df['batter_name'].ffill(axis=0, inplace=True) 
    
    return df

def the_count(df):
    
    df['count'] = df['balls'].map(int).map(str) +"-"+ df['strikes'].map(int).map(str)
    return df

def more_cleaning(df):
    # Change game_date dtype
    df['game_date']=pd.to_datetime(df['game_date'])
    # Apply a fillna to the bases with 0s and turn them into ints to get IDs
    df[ids] = df[ids].fillna(0)
    for i in ids:
        df[i] = df[i].map(int)
    # Binary columns
    df = pd.get_dummies(df, columns=make_binary, drop_first=True)
    df.rename(columns={'stand_R':'stand', 'p_throws_R':'p_throws'},inplace=True)
    df = pd.get_dummies(df, columns=['inning_topbot'], drop_first=True)
    df.rename(columns={'inning_topbot_Top':'inning_topbot'},inplace=True)
    # Int Conversion
    for i in integerify:
        df[i] = df[i].map(int)
    
    return df[f2]