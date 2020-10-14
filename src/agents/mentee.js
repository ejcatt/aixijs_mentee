class Mentee extends BayesAgent {
	constructor(options) {
		super(options);
		this.explore = false;
		this.bayesAgent = new BayesAgent(options);
		this.IGAgent = new KullbackLeiblerKSA(options);
		this.IGAgent.model = this.bayesAgent.model;
		this.model = this.bayesAgent.model;
		this.planner = new ExpectimaxTree(this, this.model, true);
		this.trace = MenteeTrace;
    this.mentor = new Mentor(options);
        this.eta = options.eta;
        this.max_m = options.max_m
        this.saved_dist = new Array(200);
        for (var i = 0; i < this.saved_dist.length; i++) {
            this.saved_dist[i] = new Array(this.model.C);
        }
        this.alpha = new Array(this.max_m);
        this.alpha_prob = new Array(this.max_m);
        for (var i = 1; i < this.alpha.length; i++) {
          this.alpha[i] = new Array(i-1);
          this.alpha_prob[i] = new Array(i-1);
          for (var j = 0;j<i-1;j++) {
            this.alpha[i][j] = new MenteeKullbackLeiblerKSA(options);
            this.alpha[i][j].model = this.bayesAgent.model;
            this.alpha[i][j].model.horizon = i;
            this.alpha_prob[i][j] = 1/(i*i*(i+1));
          }
        }
        this.exp_num = [0,0];
        this.alpha_sum = 0;
        this.xi_inv = new Array(200);
        for (var i = 0; i < this.xi_inv.length; i++) {
            this.xi_inv[i] = 0;
        }
	}

	selectAction(e) {
        for (var k = 0, C = this.model.C; k < C; k++) {
             this.saved_dist[this.t][k] = this.model.modelClass[k].conditionalDistribution(e);
             this.xi_inv[this.t] += this.model.weights[k] / this.saved_dist[this.t][k];
            }
        this.xi_inv[this.t] = 1/this.xi_inv[this.t];
        let rand = Math.random();
        this.explore = false;
        this.alpha_sum = 0;
        for (var i = 1; i < this.alpha.length; i++) {
          for (var j = 0;j<i-1;j++) {
              // Undo the bayesian update at time step t-j for IG agent (i,j)
               if (this.t - j > 0) {
                   var xi = 0;
                   for (var l = 0; l < this.alpha[i][j].model.C; l++) {
                         this.alpha[i][j].model.weights[l] *= this.xi_inv[this.t - j];
                        }
                   for (var l = 0, C = this.alpha[i][j].model.C; l < C; l++) {
                        if (this.alpha[i][j].model.weights[l] == 0) {
                            continue;
                           }
                        this.alpha[i][j].model.weights[l] /= this.saved_dist[this.t - j][l];
                       }
                    
                  }
               this.alpha_prob[i][j] = (1/(i*i*(i+1))) * Math.min(1,this.eta * this.alpha[i][j].planner.getValueEstimate() );
               this.alpha_sum += this.alpha_prob[i][j];
               if (this.alpha_sum > rand && !this.explore) {
                   this.explore = true;
                  }
             }
        }
        
        if (this.alpha_sum > 1) {
            console.log('Alpha_sum error');
            this.explore = false;
        } 
        
		if (this.explore) {
      console.log('Exploring');
      var mentor_action = this.mentor.selectAction(e);
      for (var i = 1; i < this.alpha.length; i++) {
        for (var j = 0;j<i-1;j++) {
          this.alpha[i][j].explore += 1;
          this.alpha[i][j].updatePolicy(mentor_action,e)
        }
      }
			return mentor_action;
		}


		return this.bayesAgent.selectAction(e);
	}

	update(a, e) {
		this.bayesAgent.update(a, e);
		this.information_gain = this.bayesAgent.information_gain;
	}
}

Mentee.params = [
    { field : 'eta', value: 1 },
    { field : 'max_m', value: 6 },
];
