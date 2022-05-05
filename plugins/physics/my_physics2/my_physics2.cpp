/*
 * File:
 * Date:
 * Description:
 * Author:
 * Modifications:
 * additional material http://www.iearobotics.com/wiki/index.php?title=Tutorial:ODE_y_robots_modulares:M%C3%B3dulo
 */

#include <ode/ode.h>
#include <plugins/physics.h>
#include <cmath>  
#include <random>
#include <time.h>

#define RADIUS 6e-2
#define L1 5e-1
#define L2 5e-1
#define H 5e-2
//#define WATER_DENSITY 1.0e3
#define BODY_DENSITY 1.07e3
#define LEG_DENSITY 0.999e3
#define GRAVITY 9.81
#define DRAG_SPHERE 0.5



static pthread_mutex_t mutex; // needed to run with multi-threaded version of ODE
/* The Geoms used in our plugin. */
static dGeomID robot_geom;
static dGeomID shank_geom;
static dGeomID thigh_geom;

static dGeomID floor_geom;
static dGeomID fluid_geom;
/* The Bodies used in our plugin. */
static dBodyID robot_body;
static dBodyID shank_body;
static dBodyID thigh_body;
static dBodyID fluid_body;
/* The Joints used in our plugin. */
static dJointID planeJointID;
/* The World in which bodies live. */
static dWorldID world; 
/* Other variables*/
static dReal buoyancy; 
static dReal shank_buoyancy; 
static dReal drag [3];
static dReal tors_drag [3];
static dReal crossArea [3];
static dMass robot_mass; 

static bool ADD_ROBOT = 1;
static bool IC = 1;
static double WATER_DENSITY = 1.0e3;
static double INITIAL_VEL = 10.0*((float) rand()/RAND_MAX)-5.0;
static double INITIAL_POS = 1.5+0.5*((float) rand()/RAND_MAX);

/*
 * Note: This plugin will become operational only after it was compiled and associated with the current world (.wbt).
 * To associate this plugin with the world follow these steps:
 *  1. In the Scene Tree, expand the "WorldInfo" node and select its "physics" field
 *  2. Then hit the [Select] button at the bottom of the Scene Tree
 *  3. In the list choose the name of this plugin (same as this file without the extention)
 *  4. Then save the .wbt by hitting the "Save" button in the toolbar of the 3D view
 *  5. Then reload the world: the plugin should now load and execute with the current simulation
 */

void webots_physics_init() {
  pthread_mutex_init(&mutex, NULL);  // needed to run with multi-threaded version of ODE

  /*
   * Here we get all the geoms associated with DEFs in Webots.
   * A Geom corresponds to the boundingObject node of the object specified
   * by the DEF name and it is required for collision detection.
   * That is why you can retreive here only DEFs of nodes which contains a
   * boundingObject node.
  */ 
  floor_geom = dWebotsGetGeomFromDEF("FLOOR");
  std::srand(time(0));
}

void AlignToZAxis(){
   /*
   * This function should prevent angular drifts along non 2D-axes.
   * It has to be called at each ODE time step
   */
   //dBodyID bodyID = mOdeBody->id();
   const dReal *rot = dBodyGetAngularVel(robot_body);
   const dReal *quat_ptr;
   dReal quat[4], quat_len;
   quat_ptr = dBodyGetQuaternion(robot_body);
   quat[0] = quat_ptr[0];
   quat[1] = 0;
   quat[2] = 0; 
   quat[3] = quat_ptr[3]; 
   quat_len = sqrt( quat[0] * quat[0] + quat[3] * quat[3] );
   quat[0] /= quat_len;
   quat[3] /= quat_len;
   dBodySetQuaternion( robot_body, quat );
   dBodySetAngularVel( robot_body, 0, 0, rot[2] );
}


void AddTorques(dBodyID body, const dReal ang_vel [3], dReal area[3],dReal drag_coeff[3]){
    for (int i = 0; i < 3; i++) { 
        if (body == robot_body){
            dReal C_ang = drag_coeff[i]*RADIUS*RADIUS*RADIUS/5.0;
            tors_drag[i] = -100.0*WATER_DENSITY*C_ang*area[i]*ang_vel[i]*std::abs(ang_vel[i]);
        //dWebotsConsolePrintf("%i : %f", i, drag[i]);   
        }else{
            dReal C_ang = drag_coeff[i]*RADIUS*RADIUS*RADIUS/5.0;
            tors_drag[i] = -WATER_DENSITY*C_ang*area[i]*ang_vel[i]*std::abs(ang_vel[i]);        }
    }
    dBodyAddRelTorque(body, tors_drag[0], tors_drag[1], tors_drag[2]);

}


void AddForces(dBodyID body, dReal Buoyancy,  const dReal velocity [3], dReal area[3],dReal drag_coeff[3]) {
    /*
    * This function add static and dynamic water related forces to the immersed robot
    */
    for (int i = 0; i < 3; i++) { 
        if (body == robot_body){
            drag[i] = -0.5*WATER_DENSITY*drag_coeff[i]*area[i]*velocity[i]*std::abs(velocity[i]);
        //dWebotsConsolePrintf("%i : %f", i, drag[i]);   
        }else{
            drag[i] = -0.5*WATER_DENSITY*drag_coeff[i]*area[i]*velocity[i]*std::abs(velocity[i]);
        }
    }
    //added mass ??
    // dBodyAddForceAtRelPos ->Add force at specified point in body in local coordinates.	
    dBodyAddForceAtRelPos(body, drag[0], Buoyancy+drag[1], drag[2], 0, 0, 0);
    //dWebotsConsolePrintf("Addding forces to robot body\n");

}


void AddInitialVel(dBodyID body) {
    // dBodyAddForceAtRelPos ->Add force at specified point in body in local coordinates.	
    //dBodyAddForceAtRelPos(body, INITIAL_IMPULSE, 0.0, 0.0, 0, 0, 0);
    //dBodySetPosition(body, 0.0, INITIAL_POS, 0.0);

    dBodySetLinearVel(body, INITIAL_VEL, 0.0, 0.0);

    //dWebotsConsolePrintf("Addding forces to robot body\n");
    IC = 0;
}

void webots_physics_step() {
  /*
   * Do here what needs to be done at every time step, e.g. add forces to bodies
   *   dBodyAddForce(body1, f[0], f[1], f[2]);
   *   ...
   */
  if (ADD_ROBOT){
     try{
       robot_geom = dWebotsGetGeomFromDEF("ROBOT");
       dWebotsConsolePrintf("ROBOT GEOM CLASS: %i",dGeomGetClass(robot_geom));
       shank_geom = dWebotsGetGeomFromDEF("SHANK");
       dWebotsConsolePrintf("SHANK GEOM CLASS: %i",dGeomGetClass(shank_geom));
       thigh_geom = dWebotsGetGeomFromDEF("THIGH");
       dWebotsConsolePrintf("THIGH GEOM CLASS: %i",dGeomGetClass(thigh_geom));
       
       
       if (robot_geom){
          robot_body = dGeomGetBody(dSpaceGetGeom((dSpaceID)robot_geom, 0)); 
         
          dReal Volume = 4.0/3.0*M_PI*RADIUS*RADIUS*RADIUS;
          Volume = 0.2*0.1*0.4;
          for (int i = 0; i < 3; i++) {           
            crossArea [i]= M_PI*RADIUS*RADIUS;
          }
          buoyancy = GRAVITY*WATER_DENSITY*Volume;
          robot_mass.mass = BODY_DENSITY*Volume;

          dReal Inertia = robot_mass.mass/12.0;
                
          robot_mass.I[0] = Inertia*(0.2*0.2+0.1*0.1);       
          robot_mass.I[5] = Inertia*(0.2*0.2+0.4*0.4);
          robot_mass.I[10] = Inertia*(0.4*0.4+0.1*0.1);

          dMass* ptr = &robot_mass;
          dBodySetMass(robot_body, ptr);
          dWebotsConsolePrintf("Robot_Mass");	
          dWebotsConsolePrintf("%f",robot_mass.mass);	
          dBodySetAngularDamping(robot_body, 0.1);
          
       }   
       if (shank_geom){
          shank_body = dGeomGetBody(shank_geom); 
          //dWebotsConsolePrintf("shank body added to plugin");
          shank_buoyancy = GRAVITY*WATER_DENSITY*L1*H*H;
          
          robot_mass.mass = LEG_DENSITY*L1*H*H;
          dMass* ptr = &robot_mass;
          dBodySetMass(shank_body, ptr);
          //dBodySetAngularDamping(shank_body, 0.02);	
          //dWebotsConsolePrintf("shank buoyancy = %f\n",robot_mass.mass);
       }
       if (thigh_geom){
          thigh_body = dGeomGetBody(shank_geom); 
          //dWebotsConsolePrintf("thigh body added to plugin");
          dMass* ptr = &robot_mass;
          dBodySetMass(thigh_body, ptr);
          //dBodySetAngularDamping(thigh_body, 0.02);	

       }     
       
       world = dBodyGetWorld(robot_body);
       
       planeJointID = dJointCreatePlane2D(world, 0);
       dJointAttach(planeJointID, robot_body, 0);
       ADD_ROBOT = 0;

     }
     catch (...) {
       dWebotsConsolePrintf("Non existent node\n");
     }
  } else {
    /*Execute only if the robot has been created */
    AlignToZAxis();  
    //dReal Area[3] = {crossArea,crossArea,crossArea};
    dReal Coeff[3] = {DRAG_SPHERE,DRAG_SPHERE,DRAG_SPHERE};    
    const dReal* ptr_vel = dBodyGetLinearVel(robot_body);
    const dReal* ptr_angvel = dBodyGetAngularVel(robot_body);
    if (IC){
        AddInitialVel(robot_body);
    }
    //AddTorques(robot_body, ptr_angvel, Area, Coeff); 
    //AddTorques(thigh_body, ptr_angvel, Area, Coeff); 
    //AddTorques(robot_body, ptr_angvel, Area, Coeff); 

    AddForces(shank_body, shank_buoyancy, ptr_vel, crossArea, Coeff);    
    AddForces(thigh_body, shank_buoyancy, ptr_vel, crossArea, Coeff);    
    AddForces(robot_body, buoyancy, ptr_vel, crossArea, Coeff); 

  }
  //dWebotsConsolePrintf("Running physics plugin");
  //const dReal* fluid_vel = dBodyGetPosition(fluid_body);
}

int webots_physics_collide(dGeomID g1, dGeomID g2) {
  /*
   * This function needs to be implemented if you want to overide Webots collision detection.
   * It must return 1 if the collision was handled and 0 otherwise.
   * Note that contact joints should be added to the contact_joint_group which can change over the time, e.g.
   *   n = dCollide(g1, g2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));
   *   dJointGroupID contact_joint_group = dWebotsGetContactJointGroup();
   *   dWorldID world = dBodyGetWorld(body1);
   *   ...
   *   pthread_mutex_lock(&mutex);
   *   dJointCreateContact(world, contact_joint_group, &contact[i])
   *   dJointAttach(contact_joint, body1, body2);
   *   pthread_mutex_unlock(&mutex);
   *   ...
   */
   /*we probably need to do it once we have our specific sensor set*/
   /*
   * For the collisions, we allow up to 10 contact points,
   * this is probably overkilled.
   * NOTA: la funzione non deve essere chiamata esplicitamente
   */
  dContact contact[3];
  int i, n;
  if ((dAreGeomsSame(g1, floor_geom) && dSpaceQuery((dSpaceID)robot_geom, g2) == 1) ||
             (dSpaceQuery((dSpaceID)robot_geom, g1) == 1 && dAreGeomsSame(g2, floor_geom))) {
    n = dCollide(g1, g2, 10, &contact[0].geom, sizeof(dContact));
    if (n == 0)
      return 1;
    dBodyID body = dGeomGetBody(g1);
    if (body == NULL)
      body = dGeomGetBody(g2);
    if (body == NULL)
      return 0;
    dWorldID world = dBodyGetWorld(body);
    dJointGroupID contact_joint_group = dWebotsGetContactJointGroup();
     /*
     * Totally non bouncy collision
     * For more information on the possible parameters, please
     * refer to the ODE documenatation.
     */
    for (i = 0; i < n; i++) {
      contact[i].surface.mode = dContactMu2 | dContactBounce | dContactApprox1 | dContactSoftCFM;
      contact[i].surface.mu = 0.4;
      contact[i].surface.mu2 = 0.8;
      contact[i].surface.bounce = 0.1;
      contact[i].surface.bounce_vel = 0.1;
      contact[i].surface.soft_cfm = 0.001;

      /* We add these points to the simulation. */
      pthread_mutex_lock(&mutex);
      dJointAttach(dJointCreateContact(world, contact_joint_group, &contact[i]), robot_body, NULL);
      pthread_mutex_unlock(&mutex);
    }
    return 1;

    /* We have not handled the collision. */
  } else
    return 0;
}

void webots_physics_cleanup() {
  //dGeomDestroy(robot_geom);
  ADD_ROBOT = 1;
  pthread_mutex_destroy(&mutex);
  dWebotsConsolePrintf("Clean up\n");
  //WATER_DENSITY = 1.0e3+ (10.0*((float) rand()/RAND_MAX)-5.0);
  //INITIAL_VEL = 0.0; //+10.0*((float) rand()/RAND_MAX)-5.0;
  IC = 1;
  //dWebotsConsolePrintf("The water desity at next episode will be:\n");
  //dWebotsConsolePrintf("%f\n",WATER_DENSITY); 
  //intial condition for the robot initial velocity (aggiungiamo una forza solo per la prima iterazione, un impulso)
  //mentre la quota andrà aggiornata da SUpervisor
}
