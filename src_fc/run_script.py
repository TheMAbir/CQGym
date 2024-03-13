if __name__ == '__main__':
    #编写训练命令
    for alg_name in ['FCFS','DQL','PG','A2C','PPO']:
        print('python cqsim.py -j train.swf -n train.swf -R 1500 --is_training 1 '+'--on_cuda cuda:0'+' --output_weight_file '+alg_name+'0'+' --rl_alg '+alg_name+' --debug '+'train'+alg_name+'0')
        print('python cqsim.py -j train.swf -n train.swf -r 1501 -R 1500 --is_training 1 '+'--on_cuda cuda:0'+' --input_weight_file '+alg_name+'0'+' --output_weight_file '+alg_name+'1'+' --rl_alg '+alg_name+' --debug '+'train'+alg_name+'1')
    #编写验证命令
    for alg_name in ['DQL','PG','A2C','PPO']:
        print('python cqsim.py -j validate.swf -n validate.swf -R 5000 --is_training 0 --on_cuda cuda:0 --input_weight_file '+alg_name+'0'+' --rl_alg '+alg_name+' --debug '+'valid'+alg_name+'0')
        print('python cqsim.py -j validate.swf -n validate.swf -R 5000 --is_training 0 --on_cuda cuda:0 --input_weight_file '+alg_name+'1'+' --rl_alg '+alg_name+' --debug '+'valid'+alg_name+'1')
    print('python cqsim.py -j validate.swf -n validate.swf -R 5000 --is_training 0 --rl_alg FCFS'+' --debug '+'validFCFS')