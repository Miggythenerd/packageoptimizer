                                               O o  o O  o  o  o  o o
  _____     ____    _____    ___       _     _     ____    _____     o
 |  __ \   | ___|  /     \  |   \     | \   / |   | ___|   \    \   O
 |     /   | __|   |  -  |  | |) |    |  \_/  |   | __|    / [] |    o
 |__|__\   |____|  |__|__|  |___/     |_|   |_|   |____|   |    |___][_
 _______   ______  _______  _____     _________   ______   |___________\
 \_____/---\____/--\_____/--\___/->->-\_______/->-\____/->-|___________/
  O   O     O  O    O   O    O O       O  O  O     O  O     O   O   O O\


  > > > Container Packing Visualizer < < <

This is a Python container packing program that tries to cram rectangular items into a rectangular container as efficiently as possible, then shows the result visually.

You can view the result from multiple angles, which makes it a lot easier to see where the space actually went.


> Input format:
   For containers,
       (width, depth, height)
   For items,
       (width, depth, height, amount, label)


> How the algorithm works
  Start with one big empty space (the container)
  Sort items by volume (big items first)
  For each item:
  Try all rotations
  Try all free spaces
  Score each possible placement
  Pick the best one
  Split the used space into smaller free spaces
  Merge free spaces when possible
  Repeat until no more items fit

Explore around I guess
The exe is located in the "dist" folder
