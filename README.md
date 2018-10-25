# Teaching a Robot to walk

This python file implements the **Augmented Random Search** algorithm. I did it as a midterm assignment for [Move37](https://www.theschool.ai/courses/move-37-course/), a [School of AI](https://www.theschool.ai) course.


## Getting started

Run the file to start training. The hyperparameters (including number of training steps) can be tuned in  the ``ARS_agent`` class.


### Prerequisites

You need to install both ``gym`` and ``Box2D`` to run the file.

```
pip install gym
pip install box2d box2d-kengz --user
```


## Result

After 1000 iterations of the algorithm, our robolt walks, and even runs !

![alt text](https://github.com/RomDeffayet/Bipedal_robot_walk/blob/master/results_1000steps.gif)


## License

This project is under the GNU General Public License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

Inspired from [Colin Skow](https://github.com/colinskow)'s [implementation](https://github.com/colinskow/move37/tree/master/ars)
