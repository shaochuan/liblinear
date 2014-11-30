#include <unordered_map>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <proto/model.pb.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
  printf(
  "Usage: train [options] training_set_file [model_file]\n"
  "options:\n"
  "-s type : set type of solver (default 1)\n"
  "  for multi-class classification\n"
  "  0 -- L2-regularized logistic regression (primal)\n"
  "  1 -- L2-regularized L2-loss support vector classification (dual)\n"
  "  2 -- L2-regularized L2-loss support vector classification (primal)\n"
  "  3 -- L2-regularized L1-loss support vector classification (dual)\n"
  "  4 -- support vector classification by Crammer and Singer\n"
  "  5 -- L1-regularized L2-loss support vector classification\n"
  "  6 -- L1-regularized logistic regression\n"
  "  7 -- L2-regularized logistic regression (dual)\n"
  "  for regression\n"
  " 11 -- L2-regularized L2-loss support vector regression (primal)\n"
  " 12 -- L2-regularized L2-loss support vector regression (dual)\n"
  " 13 -- L2-regularized L1-loss support vector regression (dual)\n"
  "-c cost : set the parameter C (default 1)\n"
  "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
  "-e epsilon : set tolerance of termination criterion\n"
  " -s 0 and 2\n"
  "   |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
  "   where f is the primal function and pos/neg are # of\n"
  "   positive/negative data (default 0.01)\n"
  " -s 11\n"
  "   |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
  " -s 1, 3, 4, and 7\n"
  "   Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
  " -s 5 and 6\n"
  "   |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
  "   where f is the primal function (default 0.01)\n"
  " -s 12 and 13\n"
  "   |f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
  "   where f is the dual function (default 0.1)\n"
  "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
  "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
  "-v n: n-fold cross validation mode\n"
  "-q : quiet mode (no outputs)\n"
  );
  exit(1);
}

void exit_input_error(int line_num)
{
  fprintf(stderr,"Wrong input format at line %d\n", line_num);
  exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
  int len;

  if (fgets(line,max_line_len,input) == NULL)
    return NULL;

  while (strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line,max_line_len);
    len = (int) strlen(line);
    if (fgets(line+len,max_line_len-len,input) == NULL)
      break;
  }
  return line;
}

void parse_command_line(int argc, char const *argv[], char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct feature_node *x_space;
svm::model::SolverContext context;
struct problem prob;
svm::model::Model* model_;
int flag_cross_validation;
int nr_fold;
double bias;

int main(int argc, char const *argv[])
{
  char input_file_name[1024];
  char model_file_name[1024];
  // const char *error_msg;

  parse_command_line(argc, argv, input_file_name, model_file_name);
  read_problem(input_file_name);
  // error_msg = check_parameter(&prob,&param);

  // if (error_msg)
  // {
  //   fprintf(stderr,"ERROR: %s\n",error_msg);
  //   exit(1);
  // }

  if (flag_cross_validation)
  {
    do_cross_validation();
  }
  else
  {
    model_ = train(&prob, context);
      std::fstream output(model_file_name, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!model_->SerializeToOstream(&output)) {
      std::cerr << "can't save model to file " << model_file_name << std::endl;
      exit(1);
    }
  }
  free(prob.y);
  free(prob.x);
  free(x_space);
  free(line);

  return 0;
}

void do_cross_validation()
{
  int i;
  int total_correct = 0;
  double total_error = 0;
  double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
  double *target = Malloc(double, prob.l);

  cross_validation(&prob, context, nr_fold, target);
  if (context.solver_type() == svm::model::L2R_L2LOSS_SVR ||
     context.solver_type() == svm::model::L2R_L1LOSS_SVR_DUAL ||
     context.solver_type() == svm::model::L2R_L2LOSS_SVR_DUAL)
  {
    for (i=0;i<prob.l;i++)
    {
      double y = prob.y[i];
      double v = target[i];
      total_error += (v-y)*(v-y);
      sumv += v;
      sumy += y;
      sumvv += v*v;
      sumyy += y*y;
      sumvy += v*y;
    }
    printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
    printf("Cross Validation Squared correlation coefficient = %g\n",
        ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
        ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
        );
  }
  else
  {
    for (i=0;i<prob.l;i++)
      if (target[i] == prob.y[i])
        ++total_correct;
    printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
  }

  free(target);
}

void parse_command_line(int argc, char const *argv[], char *input_file_name, char *model_file_name)
{
  int i;
  void (*print_func)(const char*) = NULL; // default printing to stdout

  context.set_eps(INF); // see setting below
  flag_cross_validation = 0;
  bias = -1;

  // parse options
  for (i=1;i<argc;i++)
  {
    if (argv[i][0] != '-') break;
    if (++i>=argc)
      exit_with_help();
    switch(argv[i-1][1])
    {
      case 's':
        context.set_solver_type(svm::model::SolverType(atoi(argv[i])));
        break;

      case 'c':
        context.set_c(atof(argv[i]));
        break;

      case 'p':
        context.set_p(atof(argv[i]));
        break;

      case 'e':
        context.set_eps(atof(argv[i]));
        break;

      case 'B':
        bias = atof(argv[i]);
        break;

      case 'w':
        context.add_weight_label(atoi(&argv[i-1][2]));
        context.add_weight(atof(argv[i]));
        break;

      case 'v':
        flag_cross_validation = 1;
        nr_fold = atoi(argv[i]);
        if (nr_fold < 2)
        {
          fprintf(stderr,"n-fold cross validation: n must >= 2\n");
          exit_with_help();
        }
        break;

      case 'q':
        print_func = &print_null;
        i--;
        break;

      default:
        fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
        exit_with_help();
        break;
    }
  }

  set_print_string_function(print_func);

  // determine filenames
  if (i>=argc)
    exit_with_help();

  strcpy(input_file_name, argv[i]);

  if (i<argc-1)
    strcpy(model_file_name,argv[i+1]);
  else
  {
    const char *p = strrchr(argv[i],'/');
    if (p==NULL)
      p = argv[i];
    else
      ++p;
    sprintf(model_file_name,"%s.model",p);
  }

  if (context.eps() == INF)
  {
    switch(context.solver_type())
    {
      case svm::model::L2R_LR:
      case svm::model::L2R_L2LOSS_SVC:
        context.set_eps(0.01);
        break;
      case svm::model::L2R_L2LOSS_SVR:
        context.set_eps(0.001);
        break;
      case svm::model::L2R_L2LOSS_SVC_DUAL:
      case svm::model::L2R_L1LOSS_SVC_DUAL:
      case svm::model::MCSVM_CS:
      case svm::model::L2R_LR_DUAL:
        context.set_eps(0.1);
        break;
      case svm::model::L1R_L2LOSS_SVC:
      case svm::model::L1R_LR:
        context.set_eps(0.01);
        break;
      case svm::model::L2R_L1LOSS_SVR_DUAL:
      case svm::model::L2R_L2LOSS_SVR_DUAL:
        context.set_eps(0.1);
        break;
    }
  }
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
  int max_index, inst_max_index, i;
  size_t elements, j;
  FILE *fp = fopen(filename,"r");
  char *endptr;
  char *idx, *val, *label;

  if (fp == NULL)
  {
    fprintf(stderr,"can't open input file %s\n",filename);
    exit(1);
  }

  prob.l = 0;
  elements = 0;
  max_line_len = 1024;
  line = Malloc(char,max_line_len);
  while (readline(fp)!=NULL)
  {
    char *p = strtok(line," \t"); // label

    // features
    while (1)
    {
      p = strtok(NULL," \t");
      if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
        break;
      elements++;
    }
    elements++; // for bias term
    prob.l++;
  }
  rewind(fp);

  prob.bias=bias;

  prob.y = Malloc(double,prob.l);
  prob.x = Malloc(struct feature_node *,prob.l);
  x_space = Malloc(struct feature_node,elements+prob.l);

  max_index = 0;
  j=0;
  for (i=0;i<prob.l;i++)
  {
    inst_max_index = 0; // strtol gives 0 if wrong format
    readline(fp);
    prob.x[i] = &x_space[j];
    label = strtok(line," \t\n");
    if (label == NULL) // empty line
      exit_input_error(i+1);

    prob.y[i] = strtod(label,&endptr);
    if (endptr == label || *endptr != '\0')
      exit_input_error(i+1);

    while (1)
    {
      idx = strtok(NULL,":");
      val = strtok(NULL," \t");

      if (val == NULL)
        break;

      errno = 0;
      x_space[j].index = (int) strtol(idx,&endptr,10);
      if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
        exit_input_error(i+1);
      else
        inst_max_index = x_space[j].index;

      errno = 0;
      x_space[j].value = strtod(val,&endptr);
      if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
        exit_input_error(i+1);

      ++j;
    }

    if (inst_max_index > max_index)
      max_index = inst_max_index;

    if (prob.bias >= 0)
      x_space[j++].value = prob.bias;

    x_space[j++].index = -1;
  }

  if (prob.bias >= 0)
  {
    prob.n=max_index+1;
    for (i=1;i<prob.l;i++)
      (prob.x[i]-2)->index = prob.n;
    x_space[j-2].index = prob.n;
  }
  else
    prob.n=max_index;

  fclose(fp);
}
