/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine generator;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);
    p.weight = 1;
    particles.push_back(p);
    weights.push_back(p.weight);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine generator;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i=0; i<num_particles; i++) {
    double theta = particles[i].theta;
    particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta)) + dist_x(generator);
    particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t)) + dist_y(generator);
    particles[i].theta += delta_t * yaw_rate + dist_theta(generator);
  }

}

double distance(LandmarkObs obs1, LandmarkObs obs2) {
  return sqrt(pow(obs1.x - obs2.x, 2) + pow(obs1.y - obs2.y, 2));
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i=0; i<predicted.size(); i++) {
    int idx = 0;
    double min_dist = std::numeric_limits<double>::infinity();
    for (int j=0; j<observations.size(); j++) {
      double dist = distance(predicted[i], observations[j]);
      if (dist < min_dist) {
        min_dist = dist;
        idx = j;
      }
    }
    predicted[i] = observations[idx];
  }
  for (int i=0; i<observations.size(); i++) {
    observations[i] = predicted[i];
  }
}

double multivar_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
  return 1 / (2 * M_PI * sig_x * sig_y) * exp(-(pow(x - mu_x, 2) / (2*sig_x*sig_x) + pow(y - mu_y, 2) / (2*sig_y*sig_y)));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // For each particle
  // 0. Convert Map to LandmarkObs
  // 1. Translate observations from Vehicle CS (coord sys) to Map CS
  // 2. Get associations between map_landmarks and observations (in Map CS)
  // 3. For each association calc probability (using multi-variate norm dist)
  
  vector<LandmarkObs> m_landmarks;
  for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
    LandmarkObs lm_obs;
    lm_obs.id = map_landmarks.landmark_list[i].id_i;
    lm_obs.x = map_landmarks.landmark_list[i].x_f;
    lm_obs.y = map_landmarks.landmark_list[i].y_f;
    m_landmarks.push_back(lm_obs);
  }
  
  for (int i=0; i<num_particles; i++) {
    // 1. Translate observations from Vehicle CS (coord sys) to Map CS
    vector<LandmarkObs> obs_landmarks;
    double theta = particles[i].theta;
    double c = cos(theta);
    double s = sin(theta);
    for (int j=0; j<observations.size(); j++) {
      LandmarkObs obs_lm;
      obs_lm.id = observations[j].id;
      obs_lm.x = c * observations[j].x - s * observations[j].y + particles[i].x;
      obs_lm.y = s * observations[j].x + c * observations[j].y + particles[i].y;
      obs_landmarks.push_back(obs_lm);
    }
    // 2. Get associations between map_landmarks and observed landmarks (in Map CS)
    dataAssociation(m_landmarks, obs_landmarks);
    // 3. For each association calc probability (using multi-variate norm dist)
    double weight = 1.0;
    for (int j=0; j<observations.size(); j++) {
      double prob = multivar_normal(
                                    obs_landmarks[j].x, obs_landmarks[j].y,
                                    m_landmarks[j].x, m_landmarks[j].y,
                                    std_landmark[0], std_landmark[1]);
      weight *= prob;
    }
    particles[i].weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> new_particles;
  for (int i=0; i<num_particles; i++) {
    weights[i] = particles[i].weight;
  }
  for(int i=0; i<num_particles; ++i) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
