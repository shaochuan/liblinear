#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include <memory>
#include <proto/model.pb.h>

struct feature_node
{
  int index;
  double value;
};

struct problem
{
  int l, n;
  double *y;
  struct feature_node **x;
  double bias;            /* < 0 if no bias term */  
};

// enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

// struct parameter
// {
//   int solver_type;

//   /* these are for training only */
//   double eps;         /* stopping criteria */
//   double C;
//   int nr_weight;
//   int *weight_label;
//   double* weight;
//   double p;
// };

// struct model
// {
//   struct parameter param;
//   int nr_class;   /* number of classes */
//   int nr_feature;
//   double *w;
//   int *label;   /* label of each class */
//   double bias;
// };

svm::model::Model* train(const struct problem *prob, const svm::model::SolverContext& param);
void cross_validation(const struct problem *prob, const svm::model::SolverContext& param, int nr_fold, double *target);

double predict_values(const svm::model::Model& model_, const struct feature_node *x, double* dec_values);
double predict(const svm::model::Model& model_, const struct feature_node *x);
double predict_probability(const svm::model::Model& model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const svm::model::Model& model_);
// svm::model::Model* load_model(const char *model_file_name);

// int get_nr_feature(const svm::model::Model& model_);
// int get_nr_class(const svm::model::Model& model_);
// void get_labels(const svm::model::Model& model_, int* label);
// double get_decfun_coef(const svm::model::Model& model_, int feat_idx, int label_idx);
// double get_decfun_bias(const svm::model::Model& model_, int label_idx);

// void free_model_content(svm::model::Model& model_ptr);
// void free_and_destroy_model(svm::model::Model& *model_ptr_ptr);
// void destroy_param(svm::model::SolverContext& param);

const char *check_parameter(const struct problem *prob, const svm::model::SolverContext& param);
int check_probability_model(const svm::model::Model& model);
int check_regression_model(const svm::model::Model& model);
void set_print_string_function(void (*print_func) (const char*));


#endif /* _LIBLINEAR_H */

