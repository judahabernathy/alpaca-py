prechecks:
  session_open_or_ext: "If closed, report next session with tz; extended-hours allowed if specified."
batch_screen:
  output_table_cols: ["symbol","close","sma50","atr","pir_5d_pct","rvol15","candidates","confirm"]
  confirmation_gate:
    rvol15_min: 1.2
  # Map classifier neutral output to stand_down for strategy modules
  neutral_label: "stand_down"
