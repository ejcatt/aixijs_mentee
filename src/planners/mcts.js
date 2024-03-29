class ExpectimaxTree {
	constructor(agent, model, nocaching) {
		this.model = model;
		this.horizon = agent.horizon;
		this.ucb = agent.ucb;
		this.max_reward = agent.max_reward;
		this.min_reward = agent.min_reward;
		this.rew_range = Math.abs(this.max_reward - this.min_reward);
		this.numActions = agent.numActions;
		this.samples = agent.samples;
		this.timeout = agent.timeout;
		if (this.timeout) {
			this.samples_total = 0;
		}
		this.gamma = agent.gamma;
		this.agent = agent; // TODO fix
		if (nocaching) {
			this.prune = this.reset;
		}

		this.reset();
	}

	getValueEstimate() {
		if (!this.sampled) {
			this.model.save();
			if (this.timeout) {
				// time budget
				var t0 = performance.now();
				var n = 0;
				while (performance.now() - t0 < this.timeout) {
					this.root.sample(this, 0);
					this.model.load();
					n++;
				}
				this.samples_total += n;
			} else {
				// sample budget
				for (let iter = 0; iter < this.samples; iter++) {
					this.root.sample(this, 0);
					this.model.load();
				}
			}

			this.sampled = true;
		}

		return ((this.root.mean / this.horizon) - this.min_reward) / this.rew_range;
	}

	bestAction() {
		this.getValueEstimate();
		return Util.argmax(this.root, (n, a) => {
			let child = n.getChild(a);
			return child ? child.mean : this.min_reward; // changed from 0 to min reward
		}, this.numActions);
	}

	getPlan() {
		let current = this.root;
		let ret = [];
		while (current) {
			let a = Util.argmax(current, (n, a) => {
				let child = n.getChild(a);
				return child ? child.mean : this.min_reward; // changed from 0 to min reward
			}, this.numActions);

			ret.push(a);
			let chanceNode = current.getChild(a);

			if (!chanceNode) {
				return ret;
			}

			let child = null;
			let maxVisits = 0;
			for (let [key, val] of chanceNode.children) {
				if (val.visits > maxVisits) {
					child = val; //No tie-breaking for now
					maxVisits = val.visits;
				}
			}

			current = child;
		}

		return ret;
	}

	rollout(horizon, dfr) {
		var reward = 0;
		for (var i = dfr; i <= horizon; i++) {
			var action = Math.floor(Math.random() * this.numActions);
			this.model.perform(action);
			var e = this.model.generatePercept();
			this.model.bayesUpdate(action, e);
			reward += this.agent.reward(e, i);
		}

		return reward;
	}

	reset() {
		let agent = this.agent;
		this.rew_range = agent.discount(0, agent.t) * Math.abs(agent.max_reward - agent.min_reward);
		this.root = new DecisionNode(null, this);
		this.sampled = false;
	}

	prune(a, e) {
		let cn = this.root.getChild(a);
		if (!cn) {
			return this.reset();
		}

		this.root = cn.getChild(e, this);
		if (!this.root) {
			return this.reset();
		}

		this.sampled = false;
	}
}

class DecisionNode {
	constructor(e, tree) {
		this.visits = 0;
		this.mean = 0;
		this.e = e;
		this.children = new Array(tree.numActions);
		this.n_children = 0;
		this.U = Util.randInts(tree.numActions);
	}

	addChild(a) {
		this.children[a] = new ChanceNode(a);
	}

	getChild(a) {
		return this.children[a];
	}

	selectAction(tree, dfr) {
		let a;
		if (this.n_children != tree.numActions) {
			a = this.U[this.n_children];
			this.addChild(a);
			this.n_children++;
		} else {
			let max = Number.NEGATIVE_INFINITY;
			for (let action = 0, A = tree.numActions; action < A; action++) {
				let child = this.getChild(action);
				let normalization = (tree.horizon - dfr + 1) * tree.rew_range;
				let value = child.mean / normalization + tree.ucb *
					Math.sqrt(Math.log2(this.visits) / child.visits);
				if (value > max) {
					max = value;
					a = action;
				}
			}
		}

		return a;
	}

	sample(tree, dfr) {
		let reward = 0;
		if (dfr > tree.horizon) {
			return 0;
		} else if (this.visits == 0) {
			reward = 0; //tree.rollout(tree.horizon, dfr);
		} else {
			let action = this.selectAction(tree, dfr);
			reward = this.getChild(action).sample(tree, dfr);
		}

		this.mean = (1 / (this.visits + 1)) * (reward + this.visits * this.mean);
		this.visits++;
		return reward;
	}
}

class ChanceNode {
	constructor(action) {
		this.visits = 0;
		this.mean = 0;
		this.children = new Map();
		this.action = action;
	}

	addChild(e, tree) {
		this.children.set(e.obs * tree.rew_range + e.rew, new DecisionNode(e, tree));
	}

	getChild(e, tree) {
		return this.children.get(e.obs * tree.rew_range + e.rew);
	}

	sample(tree, dfr) {
		let reward = 0;
		if (dfr > tree.horizon) {
			return reward;
		} else {
			tree.model.perform(this.action);
			let e = tree.model.generatePercept();
			tree.model.bayesUpdate(this.action, e);
			if (!this.getChild(e, tree)) {
				this.addChild(e, tree);
			}

			reward = tree.agent.reward(e, dfr) + this.getChild(e, tree).sample(tree, dfr + 1);
		}

		this.mean = (1 / (this.visits + 1)) * (reward + this.visits * this.mean);
		this.visits++;
		return reward;
	}
}
