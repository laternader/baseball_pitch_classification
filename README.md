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

Engineered pitch usage by grouping by pitcher name and pitch name. Then took the sum total of thrown pitches by a particular pitcher. The first was the divided by the latter thus creating...

