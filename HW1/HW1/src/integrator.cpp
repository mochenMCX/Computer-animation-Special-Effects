#include "integrator.h"

#include "configs.h"

void ExplicitEuler::integrate(const std::vector<Particles *> &particles, std::function<void(void)>) const {
  // TODO: Integrate velocity and acceleration
  //   1. Integrate velocity.
  //   2. Integrate acceleration.
  //   3. You should not compute position using acceleration. Since some part only update velocity. (e.g. impulse)
  // Note:
  //   1. You don't need the simulation function in explicit euler.
  //   2. You should do this first because it is very simple. Then you can chech your collision is correct or not.
  //   3. This can be done in 5 lines. (Hint: You can add / multiply all particles at once since it is a large matrix.)
  for (int i = 0; i < particles.size(); i++) {
    particles[i]->position() += deltaTime * particles[i]->velocity();
    particles[i]->velocity() += deltaTime * particles[i]->acceleration();
  }

}

void ImplicitEuler::integrate(const std::vector<Particles *> &particles,
                              std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Integrate velocity and acceleration using explicit euler to get Xn+1.
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.
  for (int i = 0; i < particles.size(); i++) {
    Eigen::Matrix4Xf v = particles[i]->velocity();//backup
    Eigen::Matrix4Xf a = particles[i]->acceleration(); // backup
    Eigen::Matrix4Xf p = particles[i]->position();  //backup
    // explicit euler
    particles[i]->position() += deltaTime * particles[i]->velocity();
    particles[i]->velocity() += deltaTime * particles[i]->acceleration();
    simulateOneStep(); // get new velocity and acceleration
    particles[i]->position() = p + deltaTime * particles[i]->velocity();
    particles[i]->velocity() = v + deltaTime * particles[i]->acceleration();
  }

}

void MidpointEuler::integrate(const std::vector<Particles *> &particles,
                              std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Integrate velocity and acceleration using explicit euler to get Xn+1.
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.
  for (int i = 0; i < particles.size(); i++) {
    Eigen::Matrix4Xf v = particles[i]->velocity();      // backup
    Eigen::Matrix4Xf a = particles[i]->acceleration();  // backup
    Eigen::Matrix4Xf p = particles[i]->position();      // backup
    particles[i]->position() += deltaTime / 2 * v;
    particles[i]->velocity() += deltaTime / 2 * a;
    simulateOneStep(); // get new velocity and acceleration
    particles[i]->position() = p + deltaTime * particles[i]->velocity();
    particles[i]->velocity() = v + deltaTime * particles[i]->acceleration();
  }
}

void RungeKuttaFourth::integrate(const std::vector<Particles *> &particles,
                                 std::function<void(void)> simulateOneStep) const {
  // TODO: Integrate velocity and acceleration
  //   1. Backup original particles' data.
  //   2. Compute k1, k2, k3, k4
  //   3. Compute refined Xn+1 using (1.) and (2.).
  // Note:
  //   1. Use simulateOneStep with modified position and velocity to get Xn+1.
  for (int i = 0; i < particles.size(); i++) {
    Eigen::Matrix4Xf v = particles[i]->velocity();      // backup
    Eigen::Matrix4Xf a = particles[i]->acceleration();  // backup
    Eigen::Matrix4Xf p = particles[i]->position();      // backup
    //compute k1
    Eigen::Matrix4Xf k1 = deltaTime * v;
    Eigen::Matrix4Xf k1a = deltaTime * a;
    //compute k2
    particles[i]->position() = p + k1 / 2;
    particles[i]->velocity() = v + k1a / 2;
    simulateOneStep();
    Eigen::Matrix4Xf k2 = deltaTime * particles[i]->velocity();
    Eigen::Matrix4Xf k2a = deltaTime * particles[i]->acceleration();
    //compute k3
    particles[i]->position() = p + k2 / 2;
    particles[i]->velocity() = v + k2a / 2;
    simulateOneStep();
    Eigen::Matrix4Xf k3 = deltaTime * particles[i]->velocity();
    Eigen::Matrix4Xf k3a = deltaTime * particles[i]->acceleration();
    //compute k4
    particles[i]->position() = p + k3;
    particles[i]->velocity() = v + k3a;
    simulateOneStep();
    Eigen::Matrix4Xf k4 = deltaTime * particles[i]->velocity();
    Eigen::Matrix4Xf k4a = deltaTime * particles[i]->acceleration();

    particles[i]->position() = p + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    particles[i]->velocity() = v + (k1a + 2*k2a + 2*k3a + k4) / 6;
  }
}
