'use strict';

// helper constants
const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // record info for every ball is dropped
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

/**
 * runAnalysis: Get the accuaracy of each feature independent
 */
function runAnalysis() {
  const testSetSize = 100;
  const k = 10;
  // analyze the accuaracy of each feature independent
  _.range(0, 3).forEach(feature => {
    // map the output data to work with only a feature at a time and the label
    const data = _.map(outputs, row => [row[feature], _.last(row)]);
    // get the training and test sets of data ~ N(0, 1)
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
    // get the accuaracy
    const accuaracy = _.chain(testSet)
      // get the elements where the prediction matches reality
      .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
      // get the length of the filtered array
      .size()
      // divide it with the test set length
      .divide(testSetSize)
      // get the value
      .value();
    // console.log the accuaracy
    console.log('For feature', feature, 'Accuaracy is:', accuaracy);
  });
}

/**
 * runNormalizedAnalysis: get accuaracy of normalized data for differents top 'k' values
 */
function runNormalizedAnalysis() {
  const testSetSize = 100;
  // get the training and test sets of data ~ N(0, 1)
  const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize);
  // either you can use the not normalized set of data to see what happens
  // const [testSet, trainingSet] = splitDataset(outputs, testSetSize);
  // try different values for k and get the accuaracy for that value
  _.range(1, 20).forEach(k => {
    const accuaracy = _.chain(testSet)
      // get the elements where the prediction matches reality
      .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
      // get the length of the filtered array
      .size()
      // divide it with the test set length
      .divide(testSetSize)
      // get the value
      .value();
    // console.log the accuaracy
    console.log('For k of', k, 'Accuaracy is:', accuaracy);
  });
}

/**
 * knn: K-Nearest Neighbor function
 * data: the bunch of data to analyze
 * point: the point we want to predict
 * k: the number of first 'k' elements to observe
 */
function knn(data, point, k) {
  return _.chain(data)
    // get an array of elements with:
    // -  the distance between the prediction point and the real point
    // -  the real classification for that point
    // ** IMPORTANT ** point dont have the label with it, just the features
    .map(row => [nDistance(_.initial(row), point), _.last(row)])
    // sort the mapped elements
    .sortBy(row => row[0])
    // get the first 'k' elements to analyze them
    .slice(0, k)
    // count the values of the elements (this return an object)
    .countBy(row => row[1])
    // make the object an array
    .toPairs()
    // sort the array by the counted values
    .sortBy(row => row[1])
    // get the last element (this is the higher value or the most common element)
    .last()
    // get the element analized
    .first()
    // transform it to a number
    .parseInt()
    // get the value
    .value();
}

/**
 * nDistance: get the distance between two points of n dimensions by Pythagoras Theorem (a^2 + b^2 = c^2)
 * pointA: an array with data to analyze
 * pointB: an array with data to analyze
 */
function nDistance(pointA, pointB) {
  return _.chain(pointA)
    // zip will join the dimensions from point A with the dimensions of point B
    // ej:
    // pointA = [a, b, c, d] ; pointB = [a, b, c, d]
    // zipped = [[a, a], [b, b], [c, c], [d, d]]
    .zip(pointB)
    // then for every dimension substract they values and get the square
    .map(([a, b]) => (a - b) ** 2)
    // get the sum of all values
    .sum()
    // get the root of the sum
    .value() ** 0.5;
}

/**
 * distance: get the distance beetween two points
 * pointA: the first point to analyze
 * pointB: the second point to analyze
 */
function distance(pointA, pointB) {
  // return the absolute value of the substraction between the two points
  return Math.abs(pointA - pointB);
}

/**
 * splitDataset: split the data into a training and a test set
 * data: the bunch of data to analyze
 * testCount: the length of the test data set
 */
function splitDataset(data, testCount) {
  // randomize the data order
  const shuffled = _.shuffle(data);
  // define the test data set
  const testSet = _.slice(shuffled, 0, testCount);
  // define the training data set
  const trainingSet = _.slice(shuffled, testCount);
  // return the test and the training set
  return [testSet, trainingSet];
}

/**
 * minMax: Normalize the features => feature ~ N(0, 1)
 * data: data to normalize
 * featureCount: the count of features
 */
function minMax(data, featureCount) {
  // clone the data to avoid mutate the original data
  const clonedData = _.cloneDeep(data);
  // iterate on each column (feature) we have
  // if we have 3 features, it will make 3 iterations
  for (let i = 0; i < featureCount; i++) {
    // get an array of the column values
    const column = clonedData.map(row => row[i]);
    // get min value
    const min = _.min(column);
    // get max value
    const max = _.max(column);
    // iterate on every value of the column and normalize it
    for (let j = 0; j < clonedData.length; j++) {
      // replace the value of the clonedData with the normalize value
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }
  // return the normalized clonedData
  return clonedData;
}
