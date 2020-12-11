# "STRIKE THREE! YOU'RE OUT!" But am I really?
*The app is a Work in Progress*

# Problem Statement
America's past time and, some historic games, have come to an abrupt and sour end. The largest cause of all this: umpires and their bad calls. One big offense that some umpires do that irks the players, the coaches, and the managers, are the terrible calls they make at the plate. To get a better understanding of what I mean specifically, this [video](https://youtu.be/VVf_lFixiKU) sums it all up. Umpires call balls strikes, and some throws that are right down the middle are called balls. An exaggeration but who knows, in the game of baseball, the ump's words are the final call for balls and strikes and there is no way to challenge it. If you do, you will probably be thrown out the gane which is such a shame.

In order to remedy this for the sake of players and managers, my research and model will try and determine if an umpire's ruling is as "precise" as they can be. Using Random Forests and data science, we can try to validate/rank the officiating of umpires.

## Data Collection
---
I would like to thank Mr. James LeDoux and his team/contributers for creating the [pybaseball](https://github.com/jldbc/pybaseball) (I suggest checking out there work if you ever want to work on baseball stats) Python library which made this all possible. If not for them, I would not have been able to collect the game log data of every single pitch thrown from 2015-2020, including playoffs and the World Series!

With the library, I was essentially scraping data from baseballsavant.mlb.com and gathered ~4 million rows of data with ~90 columns/features. Each row was a pitch and the characteristics of the pitch as well as minute pitcher and batter information. This dataset also included the fielders, the alignments, and the park information as well. 

In order to work with the data, I mostly used the features mentioned below. However since I was missing data on the pitcher and batters, I dropped any influence the batter would have on the pitch and looked at the unique pitch usage a pitcher would have. This made identifying pitchers unique while still trying to have their influence be within the data.

## Data Dictionary
---
Data in the dictionary are the main features used throughout majority of the models to determine strike, balls or in play.


Features:

| column name 	| dtype 	| description 	|
|-	|-	|-	|
| release_speed 	| float64 	| Pitch velocities from 2008-16 are via Pitch F/X, and adjusted to roughly out-of-hand release point. All velocities from 2017 and beyond are Statcast, which are reported out-of-hand. 	|
| release_pos_x 	| float64 	| Horizontal Release Position of the ball measured in feet from the catcher's perspective 	|
| release_pos_z 	| float64 	| Vertical Release Position of the ball measured in feet from the catcher's perspective 	|
| stand 	| uint8 	| Side of the plate batter is standing. (R = 1, L = 0) 	|
| p_throws 	| uint8 	| Hand pitcher throws with. (R = 1, L = 0) 	|
| balls 	| float64 	| Pre-pitch number of balls in count. 	|
| strikes 	| float64 	| Pre-pitch number of strikes in count. 	|
| pfx_x 	| float64 	| Horizontal movement in feet from the catcher's perspective. 	|
| pfx_z 	| float64 	| Vertical movement in feet from the catcher's perpsective. 	|
| plate_x 	| float64 	| Horizontal position of the ball when it crosses home plate from the catcher's perspective. 	|
| plate_z 	| float64 	| Vertical position of the ball when it crosses home plate from the catcher's perspective. 	|
| inning 	| int 	| Pre-pitch inning number.  	|
| inning_topbot 	| uint8 	| Pre-pitch top or bottom of inning. (Bottom = 0, Top = 1) 	|
| effective_speed 	| float64 	| Derived speed based on the the extension of the pitcher's release. 	|
| release_spin_rate 	| float64 	| Spin rate of pitch tracked by Statcast. 	|
| pitch_number 	| int 	| Total pitch number of the plate appearance. 	|
| pitch_name 	| object 	| The name of the pitch derived from the Statcast Data. 	|
| bat_score 	| int 	| Pre-pitch bat team score 	|
| fld_score 	| int 	| Pre-pitch field team score 	|
| vx0 	| float64 	| The velocity of the pitch, in feet per second, in x-dimension, determined at y=50 feet. 	|
| vy0 	| float64 	| The velocity of the pitch, in feet per second, in y-dimension, determined at y=50 feet. 	|
| vz0 	| float64 	| The velocity of the pitch, in feet per second, in z-dimension, determined at y=50 feet. 	|
| ax 	| float64 	| The acceleration of the pitch, in feet per second per second, in x-dimension, determined at y=50 feet. 	|
| ay 	| float64 	| The acceleration of the pitch, in feet per second per second, in y-dimension, determined at y=50 feet. 	|
| az 	| float64 	| The acceleration of the pitch, in feet per second per second, in z-dimension, determined at y=50 feet. 	|
| outs_when_up 	| int 	| Pre-pitch number of outs. 	|

---
The Target Variables:
| column name 	| dtype 	| description 	|
|-	|-	|-	|
| strike_attempt 	| object 	| (Y)(FE) The result of what happened to the pitch: strike, out, ball, or on-base(ob). 	|
| type 	| object 	| (Y) Short hand of pitch result. B = ball, S = strike, X = in play. 	|

\* *(FE - Feature Engineered)* 


---
Engineered Features:

*This was a way to uniquely identify pitchers without having to dummify their IDs.*

Engineered pitch usage by grouping by pitcher name and pitch name. Then took the sum total of thrown pitches by a particular pitcher. The first was the divided by the latter thus creating:

| Name 	| Definition 	|
|-	|-	|
| 2s_usage 	| Frequency of the 2-seam Fastball from 2015-2020 	|
| 4s_usage 	| Frequency of the 4-seam Fastball from 2015-2020 	|
| changeup_usage 	| Frequency of the Changeup from 2015-2020 	|
| curveball_usage 	| Frequency of the Curveball from 2015-2020 	|
| cutter_usage 	| Frequency of the Cutter from 2015-2020 	|
| eephus_usage 	| Frequency of the Eephus from 2015-2020 	|
| fastball_usage 	| Frequency of the Unspecified Fastball from 2015-2020 	|
| forkball_usage 	| Frequency of the Forkball from 2015-2020 	|
| ball_usage 	| Frequency of the Intentional Ball from 2015-2020 	|
| knucklecurve_usage 	| Frequency of the Knuckle Curve from 2015-2020 	|
| knuckleball_usage 	| Frequency of the Knuckleball from 2015-2020 	|
| pitchout_usage 	| Frequency of the Pitchout Tactic from 2015-2020 	|
| screwball_usage 	| Frequency of the Screwball from 2015-2020 	|
| sinker_usage 	| Frequency of the Sinker from 2015-2020 	|
| slider_usage 	| Frequency of the Slider from 2015-2020 	|
| split_usage 	| Frequency of the Split-Finger from 2015-2020 	|
| unknown_usage 	| Frequency of the Unknown ball from 2015-2020 	|

# EDA

There were many trends that I saw as I looked at all the data. The most intriguing was the aggregate numbers for pitch names with their release speed and release spin rate. 

![Descriptive Stats on Pitches](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/Screenshot%202020-12-10%20224922.png)

Looking at which pitch was considered a ball and strike over the past 6 seasons was intriguing. There were many outliers which explain that umpires did not do a good job making a call or the batter made a really bad decision at swinging on a bad pitch.

![balls v strikes](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/location4.png)

![balls only](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/location5.png)

What's good to know is that strikes were very consistent although it doesn't look like a very straight rectangle. It was very round.

![strikes only](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/location6.png)

Below is the straight trajectory of what a strike and ball is. If given the chance to improve this plot is to show borderline pitches that are classified wrong. That would be a stronger plot. However, this plot will contain "textbook" strikes and balls.

![strike and ball trajectory](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/movement_of_2_pitches_tv.png)

# Results 

After running a random forest on my data, these were the results I achieved:

![results](https://github.com/laternader/baseball_pitch_classification/blob/main/plots/results.png)

My results weren't as accurate as the what the data shows. I was able to surpass the baseline of 45% accuracy. However, that is pretty bad in terms of data analysis. But for sports data, it is impressive since sports has so many random events during a game that it can cause the data to contain too much noise.

We can't fire umpires but we can use this model as a stepping stone to evaluate an umpire's officiating.

## Next Steps

- Consider running a PCA model to determine the most import features to include in the model
- Explore more tuning parameters
- Determine the best (tree) model possible and other models 
- Utilize Google Colab, DataBricks, and AWS to increase the amount of computing power
- Build a model that accounts for the batter's attributes, weather, the umpires involved in the game, "clutch factors", etc.
- Start on a micro level and observe a game. Afterwards, batch all strikes and balls into one particular at-bat. Once it can group for the at-bat, then we can try and predict outs. 
