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
   * Set the number of particles. Initialize all particles to first position (based on estimates of x, y, theta
   * and their uncertainties from GPS) and all weights to 1.
   * Add random Gaussian noise to each particle.
   */
  num_particles = 300;
  std::default_random_engine generator;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  if (particles.size() != num_particles) {
    particles.resize(num_particles);
  }
  if (weights.size() != num_particles) {
    weights.resize(num_particles);
  }

  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);
    p.weight = 1;
    particles[i] = p;
    weights[i] = p.weight;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  // scale the noise (hand-tuned parameter)
  double scale = 0.1;
  for (int i=0; i<num_particles; i++) {
    double theta = particles[i].theta;
    double theta_new = theta + yaw_rate * delta_t;
    particles[i].x += velocity / yaw_rate * (sin(theta_new) - sin(theta)) + scale * dist_x(gen);
    particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta_new)) + scale * dist_y(gen);
    particles[i].theta = theta_new + scale * dist_theta(gen);
  }
}


double distBetweenObs(LandmarkObs obs1, LandmarkObs obs2) {
  return dist(obs1.x, obs1.y, obs2.x, obs2.y);
}

vector<LandmarkObs> ParticleFilter::dataAssociation(const vector<LandmarkObs>& predicted,
                                                    const vector<LandmarkObs>& observations) {
  /**
   * Assign the closest observed landmark to each predicted landmark (in Map coordinates).
   */
  vector<LandmarkObs> matched_landmarks(predicted.size());
  for (int i=0; i<predicted.size(); i++) {
    int idx = -1;
    double min_dist = std::numeric_limits<double>::infinity();
    for (int j=0; j<observations.size(); j++) {
      double d = distBetweenObs(predicted[i], observations[j]);
      if (d < min_dist) {
        min_dist = d;
        idx = j;
      }
    }
    matched_landmarks[i] = observations[idx];
  }
  return matched_landmarks;
}

double multivar_normal(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
  return 1 / (2 * M_PI * sig_x * sig_y) * exp(-(pow(x - mu_x, 2) / (2*sig_x*sig_x) + pow(y - mu_y, 2) / (2*sig_y*sig_y)));
}

LandmarkObs mapToLandmarkObs(const Map &map_landmarks, int idx) {
  LandmarkObs obs;
  obs.id = map_landmarks.landmark_list[idx].id_i;
  obs.x = map_landmarks.landmark_list[idx].x_f;
  obs.y = map_landmarks.landmark_list[idx].y_f;
  return obs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a multi-variate Gaussian distribution.
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Particles are located according to the MAP'S coordinate system.
   *   Transformation between the two systems requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // For each particle
  // 0. Convert Map object to LandmarkObs
  // 1. Translate observations from Vehicle CS (coord sys) to Map CS
  // 2. Get associations between map_landmarks and observations (in Map CS)
  // 3. For each association calc probability (using multi-variate norm dist)
  
  for (int i=0; i<num_particles; i++) {
    // 0. Convert Map object to a vector of LandmarkObs objects, discarding landmarks
    // that are beyond the sensor range
    vector<LandmarkObs> m_landmarks;
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
    for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
      LandmarkObs obs = mapToLandmarkObs(map_landmarks, j);
      if (dist(obs.x, obs.y, p_x, p_y) > sensor_range) {
        continue;
      }
      m_landmarks.push_back(obs);
    }

    // 1. Translate observations from Vehicle CS (coord sys) to Map CS
    vector<LandmarkObs> obs_landmarks;
    double c = cos(p_theta);
    double s = sin(p_theta);
    for (int j=0; j<observations.size(); j++) {
      LandmarkObs obs;
      obs.id = observations[j].id;
      obs.x = c * observations[j].x - s * observations[j].y + p_x;
      obs.y = s * observations[j].x + c * observations[j].y + p_y;
      obs_landmarks.push_back(obs);
    }

    // 2. Get associations between map_landmarks and observed landmarks (in Map CS)
    vector<LandmarkObs> matched_landmarks = dataAssociation(m_landmarks, obs_landmarks);
    vector<int> associations(matched_landmarks.size());
    vector<double> sense_x(matched_landmarks.size());
    vector<double> sense_y(matched_landmarks.size());
    for (int j=0; j<matched_landmarks.size(); j++) {
      associations[j] = m_landmarks[j].id;
      sense_x[j] = matched_landmarks[j].x;
      sense_y[j] = matched_landmarks[j].y;
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);

    // 3. For each association calc probability (using multi-variate norm dist)
    double weight = 1.0;
    for (int j=0; j<matched_landmarks.size(); j++) {
      double prob = multivar_normal(matched_landmarks[j].x, matched_landmarks[j].y,
                                    m_landmarks[j].x, m_landmarks[j].y,
                                    std_landmark[0], std_landmark[1]);
      weight *= prob;
    }
    particles[i].weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional to their weight.
   */
  vector<Particle> new_particles;
  // Collect all particle weights into a vector
  for (int i=0; i<num_particles; i++) {
    weights[i] = particles[i].weight;
  }
  // Resample particles with probabilities proportional to their weights
  std::default_random_engine gen;
  std::discrete_distribution<int> disc_dist(weights.begin(), weights.end());
  for (int i=0; i<num_particles; ++i) {
    int j = disc_dist(gen);
    new_particles.push_back(particles[j]);
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
  particle.associations = associations;
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
