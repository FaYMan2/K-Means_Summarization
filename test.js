
const data = [
    [1,1],[0,1],[1,0],
    [10,10],[10,13],[13,13],
    [54,54],[55,55],[89,89],[57,55]
]

const skmeans = require("skmeans");
const math = require('mathjs')
const res = skmeans(data,3);

console.log(data.map( (value) => {
    return math.norm(value)
}))
