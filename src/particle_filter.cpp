/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

//using namespace std; // this is poor practice, namespace pollution!!!

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // clear out the containers
  particles.clear();
  weights.clear();

  num_particles = 10;
  std::default_random_engine random_gen; // random number generator

  // create Gaussian distributions
  std::normal_distribution<double> distribution_x    (x, std[0]);
  std::normal_distribution<double> distribution_y    (y, std[1]);
  std::normal_distribution<double> distribution_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++)
  {
    Particle p; // create a particle

    // set particle's parameters
    p.x = distribution_x(random_gen);
    p.y = distribution_y(random_gen);
    p.theta = distribution_theta(random_gen);
    p.weight = 1.0;

    particles.push_back(p); // add to container
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  double epsilon = 0.005; // 0.005 [rad/sec] ~= 0.29 [deg/sec]

  std::default_random_engine random_gen; // random number generator

  /* NOTE: The lecture says we'll add noise to the velocity and yaw rate. However, the
   *       starter code is setup to add noise to the resultant state (x,y,theta) instead !
   */

  // create Gaussian distributions
  std::normal_distribution<double> distribution_x    (0.0, std_pos[0]);
  std::normal_distribution<double> distribution_y    (0.0, std_pos[1]);
  std::normal_distribution<double> distribution_theta(0.0, std_pos[2]);

  // loop thru all the particles
  for (int i=0; i<num_particles; i++) {
    // create local variables for code clarity
    double x, y, theta;
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;

    if (fabs(yaw_rate) < epsilon) {
      // straight line motion
      theta = theta_p;
      x = x_p + velocity*delta_t*cos(theta_p);
      y = y_p + velocity*delta_t*sin(theta_p);
    } else {
      // curved motion
      theta = theta_p + yaw_rate*delta_t;
      x = x_p + velocity/yaw_rate*(sin(theta)-sin(theta_p));
      y = y_p + velocity/yaw_rate*(cos(theta_p)-cos(theta));
    }

    // add noise to resultant states
    x     += distribution_x(random_gen);
    y     += distribution_y(random_gen);
    theta += distribution_theta(random_gen);

    // assign updated states back into particle
    particles[i].x = x;
    particles[i].y = y;
    particles[i].theta = theta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  // cycle thru each observations
  int num_obs = observations.size();
  for (int no=0; no<num_obs; no++) {

    double closest_range = 999999.0;
    int closest_np = 0;

    // cycling thru each predicted
    int num_pred = predicted.size();
    for (int np=0; np<num_pred; np++) {

      // looking for closest range
      double range = dist(observations[no].x, observations[no].y, predicted[np].x, predicted[np].y);

      if ( range < closest_range) {
        closest_range = range;
        closest_np = np;
      }
    }

    // set id to id of closest
    observations[no].id = predicted[closest_np].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  int num_obs = observations.size();
  int num_landmarks = map_landmarks.landmark_list.size();
  double total_wgts = 0.0;
  weights.clear();

  // loop thru all the particles
  for (int i=0; i<num_particles; i++) {

    // create predicted observations (vector<LandmarkObs>), in map coord,
    // that are within range from the current particle.
    //    (limiting it to within range reduces the needed amount of comparisons later)
    std::vector<LandmarkObs> predicted_obs;

    for (int np=0; np<num_landmarks; np++) {
      double d = dist(particles[i].x, particles[i].y,
                      map_landmarks.landmark_list[np].x_f,
                      map_landmarks.landmark_list[np].y_f);
      if ( d <= sensor_range ) {
        LandmarkObs obs;
        obs.id = map_landmarks.landmark_list[np].id_i;
        obs.x = map_landmarks.landmark_list[np].x_f;
        obs.y = map_landmarks.landmark_list[np].y_f;
        predicted_obs.push_back(obs);
      }
    }

    // transform observations from vehicle to map coordinate, for the current particle
    std::vector<LandmarkObs> observations_map_coord;

    for (int no=0; no<num_obs; no++){
       LandmarkObs obs_map = body2map(observations[no], particles[i].x, particles[i].y, particles[i].theta);
       observations_map_coord.push_back(obs_map);
    }

    // associate observation against predicted map landmarks to get the landmark IDs
    dataAssociation(predicted_obs, observations_map_coord);

    // compute weight for each observation, & capture associated landmark (id,x,y)'s
    double wgt = 1.0;
    std::vector<int> landmark_associations;
    std::vector<double> landmark_xs;
    std::vector<double> landmark_ys;

    for (int no=0; no<num_obs; no++) {
      wgt *= probability2d(observations_map_coord[no], predicted_obs, std_landmark);
      landmark_associations.push_back(observations_map_coord[no].id);
      landmark_xs.push_back(observations_map_coord[no].x);
      landmark_ys.push_back(observations_map_coord[no].y);
    }

    // assign computed weight for each particle
    particles[i].weight = wgt;
    total_wgts += wgt;

    particles[i].associations = landmark_associations;
    particles[i].sense_x      = landmark_xs;
    particles[i].sense_y      = landmark_ys;

  } // particles loop

  // normalize particle weights
  for (int i=0; i<num_particles; i++) {
    particles[i].weight /= total_wgts;

    // also fill in the weights vector (for use later in resample() function)
    weights.push_back( particles[i].weight );
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::default_random_engine gen;
  std::vector<Particle> particles_new;

  // create distribution based on weights vector
  std::discrete_distribution<double> distribution(weights.begin(), weights.end());

  // cycling thru num_particles...
  for (int i=0; i<num_particles; i++) {
    int idx = distribution(gen); // draw from discrete_distribution

    // create a new particle and copy values over from the sampled particle.
    Particle p;
    p.x             = particles[idx].x;
    p.y             = particles[idx].y;
    p.theta         = particles[idx].theta;
    p.associations  = particles[idx].associations;
    p.sense_x       = particles[idx].sense_x;
    p.sense_y       = particles[idx].sense_y;

    // add to the new particles vector
    particles_new.push_back(p);
  }

  // replace with the new particles vector
  //  (do we need to worry about: shallow/deeep copy? memory de-allocation of old vector items?
  //    assuming not an issue since we're working with values and not reference/pointers... correct?)
  particles.clear();
  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

   return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
  std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
