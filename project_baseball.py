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
    
def batter_name(des):
    try:
        name = ' '.join(des.split(' ',2)[:2])
        return name
    except:
        return np.nan
