#include "simulation/kinematics.h"

#include <iostream>
#include "Eigen/Dense"
#include "acclaim/bone.h"
#include "util/helper.h"

#include<queue>
#include<vector>

void DFS(const acclaim::Posture& posture, acclaim::Bone* bone, std::vector<int>& visited) { 
    visited[bone->idx] = 1;
    //for start position, rotation
    if (bone->parent == nullptr) {
        bone->start_position = posture.bone_translations[bone->idx];
        bone->rotation = util::rotateDegreeZYX(posture.bone_rotations[bone->idx]);
    } else {
        bone->start_position = bone->parent->end_position;
        bone->rotation = bone->parent->rotation * bone->rot_parent_current *
                         util::rotateDegreeZYX(posture.bone_rotations[bone->idx]);
    }
    //for end position
    bone->end_position = bone->start_position + bone->rotation * (bone->dir * bone->length);
    //sibling
    acclaim::Bone* temp = bone->sibling;
    while (temp != nullptr) {
        if (!visited[temp->idx]) {
            DFS(posture, temp, visited);
        }
        temp = temp->sibling;
    }
    //child
    if (bone->child != nullptr) {
        if (!visited[bone->child->idx]) {
            DFS(posture, bone->child, visited);
        }
    }

}

namespace kinematics {

void forwardSolver(const acclaim::Posture& posture, acclaim::Bone* bone) {
    // TODO (FK)
    // You should set these variables:
    //     bone->start_position = Eigen::Vector4d::Zero();
    //     bone->end_position = Eigen::Vector4d::Zero();
    //     bone->rotation = Eigen::Matrix4d::Zero();
    // The sample above just set everything to zero
    // Hint:
    //   1. posture.bone_translations, posture.bone_rotations
    // Note:
    //   1. This function will be called with bone == root bone of the skeleton
    //   2. we use 4D vector to represent 3D vector, so keep the last dimension as "0"
    //   3. util::rotate{Degree | Radian} {XYZ | ZYX}
    //      e.g. rotateDegreeXYZ(x, y, z) means:
    //      x, y and z are presented in degree rotate z degrees along z - axis first, then y degrees along y - axis, and
    //      then x degrees along x - axis
    
    std::vector<int> visited(posture.bone_translations.size(), 0);
    
    bone->start_position = posture.bone_translations[bone->idx];
    bone->rotation = util::rotateDegreeZYX(posture.bone_rotations[bone->idx]);
    bone->end_position = bone->start_position + bone->rotation * (bone->dir * bone->length);
    visited[bone->idx] = 1;

    acclaim::Bone* temp = bone->sibling;
    while (temp != nullptr) {
        if (!visited[temp->idx]) {
            DFS(posture, temp, visited);
        }
        temp = temp->sibling;
    }
    if (bone->child) {
        DFS(posture, bone->child, visited);
    }
}

Eigen::VectorXd pseudoInverseLinearSolver(const Eigen::Matrix4Xd& Jacobian, const Eigen::Vector4d& target) {
    // TODO (find x which min(| jacobian * x - target |))
    // Hint:
    //   1. Linear algebra - least squares solution
    //   2. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Construction
    // Note:
    //   1. SVD or other pseudo-inverse method is useful
    //   2. Some of them have some limitation, if you use that method you should check it.
    Eigen::VectorXd deltatheta(Jacobian.cols());
    deltatheta.setZero();
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    deltatheta = svd.solve(target);
    return deltatheta;
}

/**
 * @brief Perform inverse kinematics (IK)
 *
 * @param target_pos The position where `end_bone` (first joint in the chain) will move to.
 * @param start_bone This bone is the last bone you can move while doing IK
 * @param posture The original AMC motion's reference, you need to modify this
 * @param jointChains A 2D vector containing multiple 1D vectors, each of which holds pointers to Eigen::Vector4d
 * constituting a chain.
 * @param boneChains A 2D vector containing multiple 1D vectors, each of which holds pointers to acclaim::Bone
 * constituting a chain.
 * @param currentBasePos The base of the current chain.
 */

bool inverseJacobianIKSolver(std::vector<Eigen::Vector4d> target_pos, acclaim::Bone* end_bone,
                             acclaim::Posture& posture, std::vector<std::vector<Eigen::Vector4d*>>& jointChains,
                             std::vector<std::vector<acclaim::Bone*>>& boneChains, Eigen::Vector4d currentBasePos) {
    constexpr int max_iteration = 1000;
    constexpr double epsilon = 1E-3; //original 1E-3
    constexpr double step = 0.1; //original 0.1
    // Since bone stores in bones[i] that i == bone->idx, we can use bone - bone->idx to find bones[0] which is the
    // root.
    acclaim::Bone* root_bone = end_bone - end_bone->idx;
    // TODO
    // Perform inverse kinematics (IK)
    // HINTs will tell you what should do in that area.
    // Of course you can ignore it (Any code below this line) and write your own code.
    acclaim::Posture original_posture(posture);

    size_t bone_num = 0;

    // Traverse each chain
    for (int chainIdx = 0; chainIdx < boneChains.size(); ++chainIdx) {      //number of bone chains
        bone_num = boneChains[chainIdx].size();
        Eigen::Matrix4Xd Jacobian(4, 3 * bone_num);
        Jacobian.setZero();

        for (int iter = 0; iter < max_iteration; ++iter) {      
            Eigen::Vector4d desiredVector = target_pos[chainIdx] - *jointChains[chainIdx][0];
            //jointChains[chainIdx][0]:     chainIdx th bonechain and first one.
            if (desiredVector.norm() < epsilon) {
                break;
            }
            // TODO (compute jacobian)
            //   1. Compute arm vectors
            //   2. Compute jacobian columns, store in `Jacobian`
            // Hint:
            //   1. You should not put rotation in jacobian if it doesn't have that DoF.
            //   2. jacobian.col(/* some column index */) = /* jacobian column */
            for (long long i = 0; i < bone_num; i++) {  //decompose the x, y, z
                Eigen::Affine3d rotation = boneChains[chainIdx][i]->rotation;
                Eigen::Vector3d arm = target_pos[chainIdx].head<3>() - boneChains[chainIdx][i]->start_position.head<3>();

                Eigen::Vector3d a1 = Eigen::Vector3d::Zero();
                Eigen::Vector3d a2 = Eigen::Vector3d::Zero();
                Eigen::Vector3d a3 = Eigen::Vector3d::Zero();
                if (boneChains[chainIdx][i]->dofrx) {
                    Eigen::Vector3d unit = rotation.matrix().col(0).head<3>();
                    Eigen::Vector3d J = unit.cross(arm);
                    Jacobian.col(3 * i) = Eigen::Vector4d(J[0], J[1], J[2], 0);
                }
                if (boneChains[chainIdx][i]->dofry) {
                    Eigen::Vector3d unit = rotation.matrix().col(1).head<3>();
                    Eigen::Vector3d J = unit.cross(arm);
                    Jacobian.col(3 * i + 1) = Eigen::Vector4d(J[0], J[1], J[2], 0);
                }
                if (boneChains[chainIdx][i]->dofrx) {
                    Eigen::Vector3d unit = rotation.matrix().col(2).head<3>();
                    Eigen::Vector3d J = unit.cross(arm);
                    Jacobian.col(3 * i + 2) = Eigen::Vector4d(J[0], J[1], J[2], 0);
                }
            }

            Eigen::VectorXd deltatheta = step * pseudoInverseLinearSolver(Jacobian, desiredVector);

            // TODO (update rotation)
            //   Update `posture.bone_rotation` (in euler angle / degrees) using deltaTheta
            // Hint:
            //   1. You can ignore rotation limit of the bone.
            // Bonus:
            //   1. You cannot ignore rotation limit of the bone.

            for (long long i = 0; i < bone_num; i++) {
                Eigen::Vector3d delta = deltatheta.segment(i * 3, 3);
                posture.bone_rotations[boneChains[chainIdx][i]->idx] +=
                    util::toDegree(Eigen::Vector4d(delta[0], delta[1], delta[2], 0));


                int index = boneChains[chainIdx][i]->idx;
                //limit of x
                if (posture.bone_rotations[index][0] < boneChains[chainIdx][i]->rxmin) {
                    posture.bone_rotations[index][0] = boneChains[chainIdx][i]->rxmin;
                } else if (posture.bone_rotations[index][0] > boneChains[chainIdx][i]->rxmax) {
                    posture.bone_rotations[index][0] = boneChains[chainIdx][i]->rxmax;
                }
                //limit of y
                if (posture.bone_rotations[index][1] < boneChains[chainIdx][i]->rymin) {
                    posture.bone_rotations[index][1] = boneChains[chainIdx][i]->rymin;
                } else if (posture.bone_rotations[index][1] > boneChains[chainIdx][i]->rymax) {
                    posture.bone_rotations[index][1] = boneChains[chainIdx][i]->rymax;
                }
                //limit of z
                if (posture.bone_rotations[index][2] < boneChains[chainIdx][i]->rzmin) {
                    posture.bone_rotations[index][2] = boneChains[chainIdx][i]->rzmin;
                } else if (posture.bone_rotations[index][2] > boneChains[chainIdx][i]->rzmax) {
                    posture.bone_rotations[index][2] = boneChains[chainIdx][i]->rzmax;
                }
            }

            forwardSolver(posture, root_bone);
            // Deal with root translation
            if (chainIdx == 0) {
                posture.bone_translations[0] =
                    posture.bone_translations[0] + (currentBasePos - *jointChains[chainIdx][bone_num]);
            }
        }
    }

    // Return whether IK is stable (i.e. whether the ball is reachable) and let the skeleton not swing its hand in the
    // air
    bool stable = true;
    for (int i = 0; i < boneChains.size(); ++i) {
        if ((target_pos[i] - *jointChains[i][0]).norm() > epsilon) {
            stable = false;
        }
    }

    // You can replace "!stable" with "false" to see unstable results, but this may lead to some unexpected outcomes.
    if (!stable) {
        posture = original_posture;
        forwardSolver(posture, root_bone);
        return false;
    } else {
        original_posture = posture;
        return true;
    }
}
}  // namespace kinematics
