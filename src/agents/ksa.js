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
		this.saved_policy_weights = new Array();
		this.explore_locations = new Array();
		this.subsets = Util.subSets([...Array(this.numActions).keys()]);
		this.ig_agents = new Array(this.subsets.length);
        for (var j = 0;j<this.subsets.length;j++) {
        	this.ig_agents[j] = new Ignore(options);
            this.ig_agents[j].ignore = this.subsets[j];
        }
	}

	selectAction(e) {
		if (this.explore_count = 0) {
			return Math.floor(Math.random() * this.numActions);
		} else {
			if (this.explore_locations.includes(e.grid_locations)) {
				return this.ig_agents[Util.sample(this.policy_weights[this.explore_locations.indexOf(e.grid_locations)])].selectAction(e);
			} else {
				return Math.floor(Math.random() * this.numActions);
			}
		}
	}

	updatePolicy(a, e) {
		// TODO change safe count to be compute ebfore for loops
		if (this.explore_locations.includes(e.grid_locations)) {
			var safe_count = 0;
			var index = this.explore_locations.indexOf(e.grid_locations);
			for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a) && this.policy_weights[index][j] != 0) {
					safe_count += 1;
				} 
			}
			for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a) && this.policy_weights[index][j] != 0) {
					this.policy_weights[index][j] = 1/safe_count;
				} else {
					this.policy_weights[index][j] = 0;
				}
			}
		} else {
			this.explore_locations.push(e.grid_locations);
			var policy_w = new Array(this.subsets.length);
			var safe_count = 0;
        	for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a)) {
					safe_count += 1;
				} 
			}
			for (var j = 0;j<this.subsets.length;j++) {
				if (!this.subsets[j].includes(a)) {
					policy_w[j] = 1/safe_count;
				} else {
					policy_w[j] = 0;
				}
			}
			this.policy_weights.push(policy_w);
		}
		var combine_saved_weights = new Array();
		var combine_weights = new Array();
		for (var j = 0;j<this.policy_weights.length;j++) {
			for (var i = 0;i<this.model.saved_weights.length;i++) {
				combine_saved_weights.push(this.model.saved_weights[i]*this.saved_policy_weights[j]);
				combine_weights.push(this.model.weights[i]*this.policy_weights[j]);
			}
		}

		this.information_gain = Util.entropy(combine_saved_weights) - Util.entropy(combine_weights);
	}

	utility(e) {
		return this.model.info_gain();
	}
}