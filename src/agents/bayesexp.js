class BayesExp extends BayesAgent {
	constructor(options) {
		super(options);
		this.explore = false;
		this.epsilon = options.epsilon;
		this.bayesAgent = new BayesAgent(options);
		this.IGAgent = new KullbackLeiblerKSA(options);
		this.IGAgent.model = this.bayesAgent.model;
		this.model = this.bayesAgent.model;
		this.planner = new ExpectimaxTree(this, this.model, true);
		this.trace = BayesExpTrace;
		this.exploring_count = 0;
	}

	selectAction(e) {
		if (this.exploring_count > 0) {
			this.exploring_count -= 1;
			return this.IGAgent.selectAction(e);
		} else {
			let V = this.IGAgent.planner.getValueEstimate();
			if (V > this.epsilon) {
				console.log('exploring')
				this.exploring_count = Math.ceil(Math.log(this.epsilon) / Math.log(0.99));
				return this.IGAgent.selectAction(e);
			} else {
				return this.bayesAgent.selectAction(e);
			}
		}
	}

	update(a, e) {
		this.epsilon *= 0.999;
		this.bayesAgent.update(a, e);
		this.information_gain = this.bayesAgent.information_gain;
	}
}

BayesExp.params = [
	{ field: 'epsilon', value: 0.99 },
];
