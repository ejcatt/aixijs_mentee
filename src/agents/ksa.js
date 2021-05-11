class SquareKSA extends BayesAgent {
	constructor(options) {
		super(options);
		this.min_reward = -1;
		this.max_reward = 0;
	}

	utility(e) {
		return -1 * this.model.xi(e);
	}
}

class ShannonKSA extends BayesAgent {
	constructor(options) {
		super(options);
		this.min_reward = 0;
		this.max_reward = 1000; // TODO fix magic no
	}

	utility(e) {
		return -1 * Math.log2(this.model.xi(e));
	}
}

class KullbackLeiblerKSA extends BayesAgent {
	constructor(options) {
		super(options);
		this.max_reward = 0;
		this.min_reward = this.model.entropy();
	}

	utility(e) {
		return this.model.info_gain();
	}
}



class MenteeKullbackLeiblerKSA extends BayesAgent {
	// The MenteeKullbackLeiblerKSA agent is a knowledge seeking agent which 
	// has a distribution over all other possible agents (possible here being agents which are uniform over some subset of all actions)
	constructor(options) {
		super(options);
		this.max_reward = 0;
		this.min_reward = this.model.entropy();
		this.explore_count = 0;
		this.policy_weights = new Array();
		this.saved_model = this.model;
		this.saved_explore_locations = new Array();
		this.explore_locations = new Array();
		this.policy_weights2 = new Array(this.model.N * this.model.N);
		this.subsets = Util.subSets([...Array(this.numActions).keys()]);
		this.ig_agents = new Array(this.subsets.length);
		this.ssamples = 5;
        for (var j = 0;j<this.subsets.length;j++) {
        	this.ig_agents[j] = new Ignore(options);
            this.ig_agents[j].ignore = this.subsets[j];
        }
        this.policy_w = new Array(this.subsets.length);
		for (var j = 0;j<this.subsets.length;j++) {
			this.policy_w[j] = 1/this.subsets.length;
		}
        for (var j = 0;j<this.model.N;j++) {
        	for (var i=0;i<this.model.N;i++) {
        		//this.policy_weights2[i*this.model.N + j] = this.policy_w;
        		this.policy_weights2[i*this.model.N + j] =  new Array(this.subsets.length);
        		for (var k = 0;k<this.subsets.length;k++) {
					this.policy_weights2[i*this.model.N + j][k] = 1/this.subsets.length;
				}
        	}
        }

        //this.saved_policy_weights = Util.arrayCopy(this.policy_weights2);
        

	}

	selectAction(e) {
		var eloc = [e.obs.x,e.obs.y];
		if (this.explore_count = 0) {
			return Math.floor(Math.random() * this.numActions);
		} else {
			if (this.explore_locations.includes(eloc)) {
				return this.ig_agents[Util.sample(this.policy_weights[this.explore_locations.indexOf(eloc)])].selectAction(e);
			} else {
				return Math.floor(Math.random() * this.numActions);
			}
		}
	}

	updatePolicy(a, e) {
		//this.saved_policy_weights = this.policy_weights;
		let eloc = e.obs.x*this.model.N + e.obs.y;
		// TODO change safe count to be compute ebfore for loops
		// console.log("e");

		// console.log(e);
		// console.log("updated policy");
		// console.log(eloc);
		
		// console.log("Eloc");
		// console.log(eloc);
		// console.log("this.exploreloc");
		// console.log(this.explore_locations);
		if (!this.explore_locations.includes(eloc)) {
			// console.log("not include eloc")
			this.explore_locations.push(eloc); //instead of e.grid_locations
			// var policy_w = new Array(this.subsets.length);
			// for (var j = 0;j<this.subsets.length;j++) {
			// 	policy_w[j] = 1/this.subsets.length;
			// }
			// // this.policy_weights.push(policy_w);
			// //this.policy_weights2[eloc] = policy_w;
			// this.policy_weights2[eloc] = [...policy_w];
			//Object.assign(this.policy_weights2,{eloc:policy_w});
		}
		
			// var index = this.explore_locations.indexOf(eloc) - 1;
			// console.log(index);

			// for (var j = 0;j<this.subsets.length;j++) {
			// 	if (!this.subsets[j].includes(a) && this.policy_weights[index][j] != 0) {
			// 		this.policy_weights[index][j] *= 1/this.subsets[j].length;
			// 	} else {
			// 		this.policy_weights[index][j] = 0;
			// 	}
			// }
			// var polsum = Util.sum(this.policy_weights[index]);
			// for (var j = 0;j<this.subsets.length;j++) {
			// 	if (!this.subsets[j].includes(a) && this.policy_weights[index][j] != 0) {
			// 		this.policy_weights[index][j] *= 1/polsum;
			// 	}
			// }

			// console.log("Policy Weight2_eloc");
			// console.log(this.policy_weights2[eloc]);

			for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a) && this.policy_weights2[eloc][j] != 0 && this.subsets[j].length != 0) {
					this.policy_weights2[eloc][j] = (this.policy_weights2[eloc][j])/this.subsets[j].length;
				} else {
					this.policy_weights2[eloc][j] = 0;
				}
			}
			var polsum = Util.sum(this.policy_weights2[eloc]);
			// console.log("polsum");
			// console.log(polsum);
			for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a) && this.policy_weights2[eloc][j] != 0) {
					this.policy_weights2[eloc][j] *= 1/polsum;
				}
			}
		// var combine_saved_weights = new Array();
		// var combine_weights = new Array();
		//var ent1 = Util.entropy(this.model.saved_weights) - Util.entropy(this.model.weights);
		//var ent1 = Util.diri_entropy(this.model.saved_weights) - Util.diri_entropy(this.model.weights);
		//console.log(ent1);
		//var ent2 = Util.entropy(this.saved_policy_weights[index]) - Util.entropy(this.policy_weights[index]);
		//console.log(Util.entropy(this.saved_policy_weights[index]));
		//console.log(this.policy_weights);
		// for (var j = 0;j<this.policy_weights.length;j++) {
		// 	for (var i = 0;i<this.model.saved_weights.length;i++) {
		// 		combine_saved_weights.push(this.model.saved_weights[i]*this.saved_policy_weights[j]);
		// 		combine_weights.push(this.model.weights[i]*this.policy_weights[j]);
		// 	}
		// }

		//this.information_gain = ent1 + ent2;

		//this.information_gain = Util.entropy(combine_saved_weights) - Util.entropy(combine_weights);
		// console.log("updated_pol_w2");
		// var finaljoke = this.policy_weights2.toString();
		// console.log(finaljoke);
	}

	value_estimate(e) {
		var val = 0;
		this.saved_model = Object.assign(Object.create(Object.getPrototypeOf(this.model)), this.model);
		this.saved_explore_locations = Util.arrayCopy(this.explore_locations);
		this.saved_policy_weights = Util.arrayCopy(this.policy_weights2);
		// console.log("saved_pol_weights");
		// var kekw = this.saved_policy_weights.toString();
		// console.log(kekw);
		// console.log("pw2_fist");
		// var joke = this.policy_weights2.toString();
		// console.log(joke);
		for (var i=1;i<=this.ssamples;i++) {
			var loc_visited = [e.grid_locations];
			var newe = e;
			for (var j=1;j<=this.horizon;j++) {
				var eloc =  newe.obs.x*this.model.N + newe.obs.y;;
				// console.log("j");
				// console.log(j);
				// console.log("eloc bot");
				// console.log(eloc);
				// console.log("epxloreloca");
				// console.log(this.explore_locations);
				// console.log("this.policy_weights2[eloc]");
				// var secjoke = this.policy_weights2[eloc].toString();
				// console.log(secjoke);

				// console.log("pw2_snd");
		  //       var jokey = this.policy_weights2.toString();
		  //       console.log(jokey);
				if (this.explore_locations.includes(eloc)) {
					// console.log("this explorloc true");
					if (this.policy_weights2[eloc].length > 0) {
						var action = this.ig_agents[Util.sample(this.policy_weights2[eloc])].selectAction(newe);
					} else {
						var action = Math.floor(Math.random() * this.numActions);
					}
				} else {
					// console.log("this explorloc false");
					var action = Math.floor(Math.random() * this.numActions);
					this.explore_locations.push(eloc);
					//this.policy_weights.push(policy_w);
					//Object.assign(this.policy_weights2,{[newe.obs.x,newe.obs.y]:policy_w});
					// this.policy_weights2[eloc] = new Array(this.subsets.length);
					//policy_w.push(this.policy_weights2);

					//Object.assign(this.policy_weights2,{eloc:policy_w});
					// console.log("policy_w_bot");
					// var jokesmoke = this.policy_weights2.toString();
					// console.log(jokesmoke);

					// console.log("policy_w");
					// console.log(this.policy_w);
					// console.log("pol_weight2_bot");
					// console.log(this.policy_weights2["0,0"]);
				}
				if (!((action <=5) && (0 <= action))) {
					action = Math.floor(Math.random() * this.numActions);
				}
				// console.log("this,model");
				// console.log(this.model);
			// 	console.log("pw2");
			// 	var jokepw2 = this.policy_weights2.toString();
			// console.log(jokepw2);
				this.model.perform(action);
				this.updatePolicy(action, newe);
				newe = this.model.generatePercept();
				this.model.bayesUpdate(action, newe);
				if (!loc_visited.includes(eloc)) {
					loc_visited.push(eloc);
				}
			}
			
// console.log("saved_w_mid");
// 			var kekwmememid = this.saved_policy_weights.toString();
// 			console.log(kekwmememid);

// 			console.log("loc_visited");
// 			console.log(loc_visited);
			val += Util.diri_entropy(this.saved_model.weights) - Util.diri_entropy(this.model.weights);
			// TODO fix, should not be 0?
			// console.log("dirient");
			// console.log(Util.diri_entropy(this.saved_model.weights) - Util.diri_entropy(this.model.weights));
			for (var j=0;j<loc_visited.length;j++) {
				var locj = loc_visited[j];
			// 	console.log("type locj");
			// 	console.log(typeof locj);
			// 	console.log("saved_w_mid1");
			// var kekwmememid1 = this.saved_policy_weights.toString();
			// console.log(kekwmememid1);
				// console.log("saved_policy_weights");
				// console.log(this.saved_policy_weights[locj]);
				// console.log("policy_weights2");
				// console.log(this.policy_weights2[locj]);
				if (typeof locj !== 'undefined') {
					if (typeof this.saved_policy_weights[locj] !== 'undefined') {
						val += Util.entropy(this.saved_policy_weights[locj]) - Util.entropy(this.policy_weights2[locj]);
						// console.log("enttop");
						// console.log(Util.entropy(this.saved_policy_weights[locj]) - Util.entropy(this.policy_weights2[locj]));
					} else {
						val += Util.entropy(this.policy_w) - Util.entropy(this.policy_weights2[locj]);
						// console.log("entbot");
						// console.log(Util.entropy(this.policy_w) - Util.entropy(this.policy_weights2[locj]));
						// console.log("entpw");
						// console.log(Util.entropy(this.policy_w));
						// console.log("ent_pw2");
						// console.log(Util.entropy(this.policy_weights2[locj]));
						
						// console.log("locj");
						// console.log(locj);
					}
				}
				
				
				
			}
			this.model = Object.assign(Object.create(Object.getPrototypeOf(this.saved_model)), this.saved_model); //Util.deepCopy(this.saved_model);
			this.policy_weights2 = Util.arrayCopy(this.saved_policy_weights);
			// console.log("saved_w_end");
			// var kekwmeme = this.saved_policy_weights.toString();
			// console.log(kekwmeme);
			// console.log("pw_endloop");
			// var jokefinalreal = this.policy_weights2.toString();
			// console.log(jokefinalreal);
			this.explore_locations = Util.arrayCopy(this.saved_explore_locations);
			
		}

		return val / this.ssamples;
	}

	utility(e) {
		//return this.model.info_gain();
		return this.information_gain;

	}
}