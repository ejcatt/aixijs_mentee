class Ignore extends BayesAgent {
	constructor(options) {
		super(options);
		this.ignore = [];
		this.safe_spaces = [];
	}


	selectAction(e) {
		this.safe_spaces = [...Array(this.numActions).keys()].filter(x => !this.ignore.includes(x));
    	if (this.safe_spaces == []) 
    		{ return 4; };
		return this.safe_spaces[Math.floor(Math.random() * this.safe_spaces.length)];
	}
}

Ignore.params = [
];
