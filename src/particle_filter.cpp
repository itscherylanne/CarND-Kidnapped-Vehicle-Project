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

//magic number to tune for particles to consider
#define NUM_PARTICLES 10
//large number to initialize dist_min for dataAssociation.
#define MAX_DIST 1E+37

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	 num_particles = NUM_PARTICLES;

	//todo: std::vector<double> weights;
	weights.resize(num_particles);
	//todo: std::vector<Particle> particles;
	particles.resize(num_particles);

	default_random_engine gen;
  double std_x = std[0], std_y = std[1], std_theta = std[2];
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);


	for(int i = 0; i < num_particles; i++)
	{
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	  particles[i].weight = 1.0;

		weights[i] = 1.0;
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;
  double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];
	normal_distribution<double> noise_x(0, std_x);
	normal_distribution<double> noise_y(0, std_y);
	normal_distribution<double> noise_theta(0, std_theta);

  //predict x, y, theta based on displacement = delta_t * change
	double x_displacement = 0.0, y_displacement = 0.0, theta_displacement = 0.0;
	for(int i=0; i < num_particles; i++)
	{
		if( fabs(yaw_rate) < 0.00001)
		{
			theta_displacement = 0.0;
			x_displacement = velocity * delta_t  * cos(particles[i].theta);
			y_displacement = velocity * delta_t  * sin(particles[i].theta);
		}
		else
		{
			theta_displacement = yaw_rate * delta_t;
			x_displacement = (velocity /yaw_rate)  * (sin(particles[i].theta + theta_displacement)-sin(particles[i].theta));
			y_displacement = (velocity /yaw_rate)  * (cos(particles[i].theta) - cos(particles[i].theta + theta_displacement));
		}
		particles[i].x += x_displacement + noise_x(gen);
		particles[i].y += y_displacement + noise_y(gen);
		particles[i].theta += theta_displacement + noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for(int i = 0; i < observations.size(); i++){
		int closest_id = -1;
		double dist_min = MAX_DIST;

		//implementation of nearest neighbor association
		//TODO: for future, may consider uncertainty estimates to weigh-in for association
		for(int j = 0; j < predicted.size(); j++){
			double dx = predicted[j].x - observations[i].x;
			double dy = predicted[j].y - observations[i].y;
			double dist2 = dx*dx + dy*dy;

			double dist_nn = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if( dist_nn < dist_min){
				dist_min = dist_nn;
				closest_id = predicted[j].id;
				//cout << "test" << endl;
			}
		}
		observations[i].id = closest_id;
	}

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	double std_x = std_landmark[0], std_y = std_landmark[1];

	for(int i = 0; i < particles.size();i++){


		//Find map_landmarks within the sensor range of a particular particle
		vector<LandmarkObs> inrange_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double dx = (particles[i].x - map_landmarks.landmark_list[j].x_f);
			double dy = (particles[i].y - map_landmarks.landmark_list[j].y_f);
      if (dx*dx+dy*dy <= sensor_range*sensor_range ) {
        inrange_landmarks.push_back(LandmarkObs {map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }

		//Transform observations from vehichle's frame to map's frame
    vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++) {
      LandmarkObs transformed_obs;
      transformed_obs.id = observations[j].id;
      transformed_obs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
      transformed_obs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
      transformed_observations.push_back(transformed_obs);
    }


		//--------------------
		// identify xformed_observations to inrange_map_landmarks using NN algorithm
		//--------------------
		dataAssociation(inrange_landmarks, transformed_observations);

		//--------------------
		//Update weight of Particle
		//--------------------
		particles[i].weight = 1.0;

		for(int j = 0; j < transformed_observations.size(); j++){
			double dx = 0; //obs - landmark
			double dy = 0;

			//calculate weight using normalization terms and exponent
			for(int k = 0; k< inrange_landmarks.size(); k++){
				 if (transformed_observations[j].id == inrange_landmarks[k].id) {
					dx = transformed_observations[j].x - inrange_landmarks[k].x;
					dy = transformed_observations[j].y - inrange_landmarks[k].y;
				}
			}//end of for(k) loop

			double gauss_norm = gauss_norm = (1.0/(2.0 * M_PI * std_x * std_y));
			double exponent = (dx*dx)/(2.0*std_x*std_x) + (dy*dy)/(2.0*std_y*std_y);
			particles[i].weight *= gauss_norm * exp(-exponent);
		}//end of for(j) loop

		weights[i] = particles[i].weight;
  }//end of for(i) loop

}

void ParticleFilter::resample() {

	default_random_engine gen;
	vector<Particle> resampled_particles;

	// uniform random sampling of all the indices
	discrete_distribution<int> indices(weights.begin(), weights.end());

/*
	// create random distribution of weights based on the max weight
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> rand_weight(0.0, max_weight);

	double beta = 0.0;
*/
	// for the total number of particles, choose a random particle
	//draw with replacement, so doubles may occur during resmaple
	for (int i = 0; i < num_particles; i++) {

		/*
		beta += rand_weight(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		*/
		int index = indices(gen);
		resampled_particles.push_back(particles[index]);
	}

	particles = resampled_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
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

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
