#include "darknet.h"
#include <assert.h>

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void train_webank_model(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int epoch, char *base)
{
    printf("start to train model\n");
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    printf("train: %s\n", train_images);
    srand(time(0));
    //char *base = basecfg(cfgfile);
    //printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    printf("current batch: %d\n", get_current_batch(net));
    int base_epoch = get_current_batch(net);
    while(get_current_batch(net) < base_epoch+epoch/*net->max_batches*/){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);



        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
		/*
		int idx = 0;
		for (idx = 0; idx < net->n; ++idx)
		{
			layer l = net->layers[idx];
			
			double sum = 0.0;
			
			int idx2 = 0;
			for (idx2 = 0; idx2 < l.nweights; ++idx2)
			{
				sum = sum + l.weights[idx2];
				
			}
			printf("layer %d: %lf\n", idx, sum);
		}
		
		printf("end!\n");
		*/
		
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_last.weights", backup_directory, base);
    save_weights(net, buff);
}


void aggregate_webank_model_weights(char *cfgfile, char *weightfile_1, char *weightfile_2, double p1, double p2, char *output_dir)
{
	printf("start to aggregate_weights\n");
	
    srand(time(0));
	printf("%s %s %s %lf %lf %s\n", cfgfile, weightfile_1, weightfile_2, &p1, &p2, output_dir);	
	network *net_total = parse_network_cfg(cfgfile);
	network *net_1 = parse_network_cfg(cfgfile);
	network *net_2 = parse_network_cfg(cfgfile);
	
	char *base = basecfg(cfgfile);
    printf("%s\n", base);

    int seed = rand();

    srand(seed);

    net_1 = load_network(cfgfile, weightfile_1, 0);
	net_2 = load_network(cfgfile, weightfile_2, 0);
	
	net_total = load_network(cfgfile, weightfile_1, 0);

	assert(net_1->n == net_2->n);
    
	
	int n_layers = net_1->n;
	
	int idx = 0;
	
	for (idx=0; idx<1; ++idx)
	{
		layer l_1 = net_1->layers[idx];
		layer l_2 = net_2->layers[idx];
		
		layer l = net_total->layers[idx];
		
		int nweights = l_1.nweights;
		int idx2=0;
		
		for (idx2=0; idx2<nweights; ++idx2)
		{
			l.weights[idx2] = 0.0;
			l.weights[idx2] = p1*l_1.weights[idx2]+p2*l_2.weights[idx2];
			
			printf("p1: %lf, p2: %lf, net1: %lf, net2: %lf, net_total: %lf\n", p1, p2, l_1.weights[idx2], l_2.weights[idx2], l.weights[idx2]);
		}
	}

    char buff[256];
    sprintf(buff, "%s/%s_final.weights", output_dir, "aggregate");
    save_weights(net_total, buff);
}

void get_model_weights(char *cfgfile, char *weightfile)
{
	//printf("start to test_aggregate\n");
	
    srand(time(0));
	
	network *net = parse_network_cfg(cfgfile);
	
	char *base = basecfg(cfgfile);
    //printf("%s\n", base);

    int seed = rand();

    srand(seed);


    net = load_network(cfgfile, weightfile, 0);

    
	
	int n_layers = net->n;
	
	int idx = 0;
	for (idx = 0; idx < net->n; ++idx)
	{
		layer l = net->layers[idx];
		
		double sum = 0.0;
		
		int idx2 = 0;
		for (idx2 = 0; idx2 < l.nweights; ++idx2)
		{
			sum = sum + l.weights[idx2];
			
		}
		printf("layer %d: %lf\n", idx, sum);
	}
	//printf("%d\n", n_layers);
}


void change_model_weights(char *cfgfile, char *weightfile, double factor, char *outputfile)
{
	
    srand(time(0));

    int seed = rand();
    srand(seed);
    network *net = load_network(cfgfile, weightfile, 0);
	network *sum = load_network(cfgfile, weightfile, 0);
	int n_layers = net->n;
	
	int idx = 0;
	for (idx=0; idx<n_layers; ++idx)
	{
		layer l = net->layers[idx];
		
		layer s = sum->layers[idx];
		
		int nweights = l.nweights;
		int idx2=0;
		
		for (idx2=0; idx2<nweights; ++idx2)
		{
			//printf("%lf ", l.weights[idx2]);
			s.weights[idx2] = l.weights[idx2]*factor;
		}
		
		printf("\n");
	}
	//printf("%d\n", n_layers);
	save_weights(sum, outputfile);
}


void run_webank_model(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
	char *weights2 = (argc > 6) ? argv[6] : 0;
    char *filename = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "train"))
	{
		char *datacfg = argv[3];
		char *cfg = argv[4];
		char *weights = (argc > 5) ? argv[5] : 0;
		
		char *epoch = (argc > 6) ? argv[6] : "5";
		
		char cur_time[256];
		sprintf(cur_time, "%d", time(0));
		
		char *output = (argc > 7) ? argv[7] : cur_time;
		train_webank_model(datacfg, cfg, weights, gpus, ngpus, clear, atoi(epoch), output);
	}
	else if(0==strcmp(argv[2], "aggregate")) 
	{
		char *cfg = argv[3];
		char *weights = (argc > 4) ? argv[4] : 0;
		char *weights2 = (argc > 5) ? argv[5] : 0;
		char *p1 = (argc > 6) ? argv[6]: 0;
		char *p2 = (argc > 7) ? argv[7]: 0;
		
		
		char *outfile = (argc > 8) ? argv[8] : "";
		aggregate_webank_model_weights(cfg, weights, weights2, atof(p1), atof(p2), outfile);
	}
	else if(0==strcmp(argv[2], "get_weights"))
	{
		char *cfg = argv[3];
		char *weights = (argc > 4) ? argv[4] : 0;
		get_model_weights(cfg, weights);
	}
	else if(0==strcmp(argv[2], "change_weights"))
	{
		char *cfg = argv[3];
		char *weights = (argc > 4) ? argv[4] : 0;
		char *p = (argc > 5) ? argv[5]: 0;
		char *outfile = (argc > 6) ? argv[6]:"";
		change_model_weights(cfg, weights, atof(p), outfile);
	}
	//else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
