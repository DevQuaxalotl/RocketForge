# Networking format

The gamestate should be sent as JSON to the renderer over its UDP socket.
The default port of RocketSimVis is `9273`.

Some JSON fields are optional, and more will be added in the future as I add features to the visualizer.

The format for the gamestate, in pseudo-JSON, is:

```
# Physics state
{ 
	"pos": [ <x>, <y>, <z> ],
	
	OPTIONAL "forward": [ <x>, <y>, <z> ], # Forward direction as a normalized vector
	OPTIONAL "up": [ <x>, <y>, <z> ], # Upward direction as a normalized vector
	# NOTE: If rotation ("forward" and "up") are not provided, 
	#       the visualizer will track its own internal rotation for the object, 
	#       and update it with "ang_vel" every frame
	
	"vel": [ <x>, <y>, <z> ],
	"ang_vel": [ <x>, <y>, <z> ]
}
```

```
{
	"ball_phys": <physics state>,
	
	"cars": [
		{ # Example car
			"team_num": <0 or 1>, # Blue = 0, orange = 1
			
			"phys": <physics state>,
			
			OPTIONAL "controls": { 
				"throttle": <v>, "seer": <v>, 
				"pitch": <v>, "yaw": <v>, "roll": <v>, 
				"boost": <v>, "jump": <v>, "handbrake": <v>, 
			},
			
			"boost_amount": <v> (from 0 to 1),
			"on_ground": <bool>,
			OPTIONAL "has_flipped_or_double_jumped": <bool>,
			"is_demoed": <bool>
		}
	],
	
	# Use this to change the locations and number of boost pads
	# If you never set this, the visualizer will use the soccar boost locations with RLGym/RLBot ordering
	OPTIONAL "boost_pad_locations": [ [<x>, <y>, <z>], [<x>, <y>, <z>], ... ],
	
	# If you don't provide this, no boost pads will be rendered
	OPTIONAL "boost_pad_states": [ <bool>, <bool>, ... ]
}
```

## RocketForge extensions

Optional fields on the render packet (trainer → viewer, port 9273):

| Field | Type | Meaning |
|---|---|---|
| `spectate_car_id` | int | Player **index** into `cars` the camera follows (not a car id). |
| `context_label` | string | Extra overlay line (e.g. the static opponent). |
| `debug` | object | Reward debug snapshot. |
| `game_speed` | float | Current render speed multiplier. |
| `paused` | bool | Whether the render loop is paused. |

Control commands (viewer → trainer, UDP JSON to `127.0.0.1:9274`):

| Command | Payload | Effect |
|---|---|---|
| `{"cmd":"focus_set","index":N}` | player index | Focus camera + reward sampling on player N (clamped). |
| `{"cmd":"speed_step","dir":1}` | +1 / -1 | Next speed preset up / down. |
| `{"cmd":"speed_reset"}` | — | Speed = 1x. |
| `{"cmd":"pause_toggle"}` | — | Pause / resume. |
