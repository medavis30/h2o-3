package hex.grid;

import hex.Model;
import hex.ModelParametersBuilderFactory;
import hex.ScoreKeeper;
import hex.ScoringInfo;
import hex.grid.HyperSpaceSearchCriteria.CartesianSearchCriteria;
import hex.grid.HyperSpaceSearchCriteria.RandomDiscreteValueSearchCriteria;
import hex.grid.HyperSpaceSearchCriteria.Strategy;
import water.exceptions.H2OIllegalArgumentException;
import water.util.PojoUtils;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public interface HyperSpaceWalker<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> {

  interface HyperSpaceIterator<MP extends Model.Parameters> {
    /**
     * Get next model parameters.
     *
     * <p>It should return model parameters for next point in hyper space.
     * Throws {@link java.util.NoSuchElementException} if there is no remaining point in space
     * to explore.</p>
     *
     * <p>The method can optimize based on previousModel, but should be
     * able to handle null-value.</p>
     *
     * @param previousModel  model generated for the previous point in hyper space, can be null.
     *
     * @return model parameters for next point in hyper space or null if there is no such point.
     *
     * @throws IllegalArgumentException  when model parameters cannot be constructed
     * @throws java.util.NoSuchElementException if the iteration has no more elements
     */
    MP nextModelParameters(Model previousModel);

    /**
     * Returns true if the iterator can continue.  Takes into account strategy-specific stopping criteria, if any.
     * @param previousModel  optional parameter which helps to determine next step, can be null
     * @return  true if the iterator can produce one more model parameters configuration.
     */
    boolean hasNext(Model previousModel);

    /**
     * Inform the Iterator that a model build failed in case it needs to adjust its internal state.
     * Implementations are expected to consume the {@code withFailedModelHyperParams} callback with the hyperParams used to create the failed model.
     * @param failedModel: the model whose training failed.
     * @param withFailedModelHyperParams: consumes the "raw" hyperparameters values used for the failed model.
     */
    void onModelFailure(Model failedModel, Consumer<Object[]> withFailedModelHyperParams);

  } // interface HyperSpaceIterator

  /**
   * Search criteria for the hyperparameter search including directives for how to search and
   * when to stop the search.
   */
  C search_criteria();

  /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
  boolean stopEarly(Model model, ScoringInfo[] sk);

  /**
   * Returns an iterator to traverse this hyper-space.
   *
   * @return an iterator
   */
  HyperSpaceIterator<MP> iterator();

  /**
   * Returns hyper parameters names which are used for walking the hyper parameters space.
   *
   * The names have to match the names of attributes in model parameters MP.
   *
   * @return names of used hyper parameters
   */
  String[] getHyperParamNames();

  /**
   * Return estimated maximum size of hyperspace, not subject to any early stopping criteria.
   *
   * Can return -1 if estimate is not available.
   *
   * @return size of hyper space to explore
   */
  long getMaxHyperSpaceSize();

  /**
   * Return initial model parameters for search.
   * @return  return model parameters
   */
  MP getParams();

  ModelParametersBuilderFactory<MP> getParametersBuilderFactory();

  /**
   * Superclass for for all hyperparameter space walkers.
   * <p>
   * The external Grid / Hyperparameter search API uses a HashMap<String,Object> to describe a set of hyperparameter
   * values, where the String is a valid field name in the corresponding Model.Parameter, and the Object is
   * the field value (boxed as needed).
   */
  abstract class BaseWalker<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> implements HyperSpaceWalker<MP, C> {

    /**
     * @see #search_criteria()
     */
    final protected C _search_criteria;

    /**
     * Search criteria for the hyperparameter search including directives for how to search and
     * when to stop the search.
     */
    public C search_criteria() { return _search_criteria; }

    /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
    @Override
    public boolean stopEarly(Model model, ScoringInfo[] sk) {
      return false;
    }

    /**
     * Parameters builder factory to create new instance of parameters.
     */
    final transient ModelParametersBuilderFactory<MP> _paramsBuilderFactory;

    /**
     * Used "base" model parameters for this grid search.
     * The object is used as a prototype to create model parameters
     * for each point in hyper space.
     */
    final MP _params;

    final MP _defaultParams;

    /**
     * Hyper space description - in this case only dimension and possible values.
     */
    final protected Map<String, Object[]> _hyperParams;

    long model_number = 0l;   // denote model number
    /**
     * Cached names of used hyper parameters.
     */
    final protected String[] _hyperParamNames;

    /**
     * Compute max size of hyper space to walk. May include duplicates if points in space are specified multiple
     * times.
     */
    final protected long _maxHyperSpaceSize;

    /**
     * Java hackery so we can have a factory method on a class with type params.
     */
    public static class WalkerFactory<MP extends Model.Parameters, C extends HyperSpaceSearchCriteria> {
      /**
       * Factory method to create an instance based on the given HyperSpaceSearchCriteria instance.
       */
      public static <MP extends Model.Parameters, C extends HyperSpaceSearchCriteria>
      HyperSpaceWalker<MP, ? extends HyperSpaceSearchCriteria> create(MP params,
                                                                      Map<String, Object[]> hyperParams,
                                                                      ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                                                                      C search_criteria) {
        Strategy strategy = search_criteria.strategy();
        switch (strategy) {
          case Cartesian:
            return new HyperSpaceWalker.CartesianWalker<>(params, hyperParams, paramsBuilderFactory, (CartesianSearchCriteria) search_criteria);
          case RandomDiscrete:
            return new HyperSpaceWalker.RandomDiscreteValueWalker<>(params, hyperParams, paramsBuilderFactory, (RandomDiscreteValueSearchCriteria) search_criteria);
          default:
            throw new H2OIllegalArgumentException("strategy", "GridSearch", strategy);
        }
      }
    }

    /**
     *
     * @param paramsBuilderFactory
     * @param hyperParams
     */
    public BaseWalker(MP params,
                      Map<String, Object[]> hyperParams,
                      ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                      C search_criteria) {
      _params = params;
      _hyperParams = hyperParams;
      _paramsBuilderFactory = paramsBuilderFactory;
      _hyperParamNames = hyperParams.keySet().toArray(new String[0]);
      _search_criteria = search_criteria;
      _maxHyperSpaceSize = computeMaxSizeOfHyperSpace();
      
      // Sanity check the hyperParams map, and check it against the params object
      try {
        _defaultParams = (MP) params.getClass().newInstance();
      } catch (Exception e) {
        throw new H2OIllegalArgumentException("Failed to instantiate a new Model.Parameters object to get the default values.");
      }
      validateParams();
    } // BaseWalker()

    @Override
    public String[] getHyperParamNames() {
      return _hyperParamNames;
    }

    @Override
    public long getMaxHyperSpaceSize() {
      return _maxHyperSpaceSize;
    }

    @Override
    public MP getParams() {
      return _params;
    }

    @Override
    public ModelParametersBuilderFactory<MP> getParametersBuilderFactory() {
      return _paramsBuilderFactory;
    }

    protected MP getModelParams(MP params, Object[] hyperParams) {
      ModelParametersBuilderFactory.ModelParametersBuilder<MP>
              paramsBuilder = _paramsBuilderFactory.get(params);
      for (int i = 0; i < _hyperParamNames.length; i++) {
        String paramName = _hyperParamNames[i];
        Object paramValue = hyperParams[i];

        if (paramName.equals("valid")) {  // change paramValue to key<Frame> for validation_frame
          paramName = "validation_frame";   // @#$, paramsSchema is still using validation_frame and training_frame
        }

        paramsBuilder.set(paramName, paramValue);
      }
      return paramsBuilder.build();
    }

    // todo: Karthik, you need to fix this if there are grouped_parameters in the hyper_parameter.  It should not
    //  be part of the grid and should not contribute to any new model.  However, you do need to use it to determine the
    //  number of legal models you can build.  Ummm, you are
    //  welcome.  Wendy
    protected long computeMaxSizeOfHyperSpace() {
      long work = 0;
      long non_grouped_params_combos = 1;
      
      // Create hashmap of number of combinations for each array length of grouped parameters
      Set<String> grouped_params = _search_criteria._grouped_parameters != null ?
              new HashSet<>(Arrays.asList(_search_criteria._grouped_parameters)) : null;
      Map<Integer, Integer> grouped_param_arrlens = null;
      if(grouped_params != null) {
        for (String param: grouped_params) {
          Map<Integer, Integer> param_arrlen_frequencies = new HashMap<>();
          if(_hyperParams.get(param) != null) {
            for (Object param_vals: _hyperParams.get(param)) {
              int param_arrlen = ((ArrayList) param_vals).toArray().length;
              if(param_arrlen_frequencies.get(param_arrlen) != null) {
                param_arrlen_frequencies.put(param_arrlen, param_arrlen_frequencies.get(param_arrlen) + 1);
              } else {
                param_arrlen_frequencies.put(param_arrlen, 1);
              }
            }
            grouped_param_arrlens = multiplyHashMaps(grouped_param_arrlens, param_arrlen_frequencies);
          }
        }
      }
      
      // Get number of combinations for non grouped parameters
      for (Map.Entry<String, Object[]> p : _hyperParams.entrySet()) {
        if(grouped_params != null && grouped_params.contains(p.getKey())) { continue; } 
        else {
          if (p.getValue() != null) {
            non_grouped_params_combos *= p.getValue().length;
          }  
        }
      }
      
      for (Map.Entry<Integer, Integer> entry : grouped_param_arrlens.entrySet()) {
        work += non_grouped_params_combos * entry.getValue();
      }
      
      return work;
    }

    protected Map<Integer, Integer> multiplyHashMaps(Map<Integer, Integer> m1, Map<Integer, Integer> m2) {
      if(m1 == null) { return m2; }
      Map<Integer, Integer> m = new HashMap<>();
      for(Map.Entry<Integer, Integer> entry : m1.entrySet()) {
        if(m2.get(entry.getKey()) != null) {
          m.put(entry.getKey(), entry.getValue() * m2.get(entry.getKey()));
        } else {
          m.put(entry.getKey(), 0);
        }
      }
      return m;
    }
    
    /** Given a list of indices for the hyperparameter values return an Object[] of the actual values. */
    protected Object[] hypers(int[] hidx) {
      Object[] hypers = new Object[_hyperParamNames.length];
      for (int i = 0; i < hidx.length; i++) {
        hypers[i] = _hyperParams.get(_hyperParamNames[i])[hidx[i]];
      }
      return hypers;
    }

    protected int integerHash(int[] ar) {
      Integer[] hashMe = new Integer[ar.length];
      for (int i = 0; i < ar.length; i++)
        hashMe[i] = ar[i] * _hyperParams.get(_hyperParamNames[i]).length;
      return Arrays.deepHashCode(hashMe);
    }

    private void validateParams() {
      // if a parameter is specified in both model parameter and hyper-parameter, this is only allowed if the
      // parameter value is set to be default.  Otherwise, an exception will be thrown.
      for (String key : _hyperParams.keySet()) {
        if (!key.equals("grouped_parameters")) {
          // Throw if the user passed an empty value list:
          Object[] values = _hyperParams.get(key);
          if (0 == values.length)
            throw new H2OIllegalArgumentException("Grid search hyperparameter value list is empty for hyperparameter: " + key);

          if ("seed".equals(key) || "_seed".equals(key)) continue;  // initialized to the wall clock

          // Ugh.  Java callers, like the JUnits or Sparkling Water users, use a leading _.  REST users don't.
          String prefix = (key.startsWith("_") ? "" : "_");

          // Throw if params has a non-default value which is not in the hyperParams map
          Object defaultVal = PojoUtils.getFieldValue(_defaultParams, prefix + key, PojoUtils.FieldNaming.CONSISTENT);
          Object actualVal = PojoUtils.getFieldValue(_params, prefix + key, PojoUtils.FieldNaming.CONSISTENT);

          if (defaultVal != null && actualVal != null) {
            // both are not set to null
            if (defaultVal.getClass().isArray() &&
                    // array
                    !PojoUtils.arraysEquals(defaultVal, actualVal)) {
              throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
            } // array
            if (!defaultVal.getClass().isArray() &&
                    // ! array
                    !defaultVal.equals(actualVal)) {
              throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
            } // ! array
          } // both are set: defaultVal != null && actualVal != null

          // defaultVal is null but actualVal is not, raise exception
          if (defaultVal == null && !(actualVal == null)) {
            // only actual is set
            throw new H2OIllegalArgumentException("Grid search model parameter '" + key + "' is set in both the model parameters and in the hyperparameters map.  This is ambiguous; set it in one place or the other, not both.");
          }
        } // for all keys
      }
    }
  }

  /**
   * Hyperparameter space walker which visits each combination of hyperparameters in order.
   */
  class CartesianWalker<MP extends Model.Parameters>
          extends BaseWalker<MP, CartesianSearchCriteria> {

    public CartesianWalker(MP params,
                           Map<String, Object[]> hyperParams,
                           ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                           CartesianSearchCriteria search_criteria) {
      super(params, hyperParams, paramsBuilderFactory, search_criteria);
    }

    @Override
    public HyperSpaceIterator<MP> iterator() {

      return new HyperSpaceIterator<MP>() {
        /** Hyper params permutation.
         */
        private int[] _currentHyperparamIndices = null;

        @Override
        public MP nextModelParameters(Model previousModel) {

          if (_currentHyperparamIndices == null) {
            _currentHyperparamIndices = new int[_hyperParamNames.length];
            _currentHyperparamIndices[0] = -1; // First combination - all 0s, may not be valid
          }
          _currentHyperparamIndices = nextModelIndices(_currentHyperparamIndices);

          if (_currentHyperparamIndices != null) {
            // Fill array of hyper-values
            Object[] hypers = hypers(_currentHyperparamIndices);
            // Get clone of parameters
            MP commonModelParams = (MP) _params.clone();
            // Fill model parameters
            MP params = getModelParams(commonModelParams, hypers);

            return params;
          } else {
            throw new NoSuchElementException("No more elements to explore in hyper-space!");
          }
        }

        @Override
        public boolean hasNext(Model previousModel) {
          if (_currentHyperparamIndices != null) {
            int[] hyperParamIndicesCopy = new int[_currentHyperparamIndices.length];
            System.arraycopy(_currentHyperparamIndices, 0, hyperParamIndicesCopy, 0, _currentHyperparamIndices.length);
            if (nextModelIndices(hyperParamIndicesCopy) == null) {
              return false;
            }
          }
          
          return true;
        }

        @Override
        public void onModelFailure(Model failedModel, Consumer<Object[]> withFailedModelHyperParams) {
          // FIXME: when using parallel grid search, there's no good reason to think that the current hyperparam indices where the ones used for the failed model
          withFailedModelHyperParams.accept(hypers(_currentHyperparamIndices));
        }

        /**
         * Cartesian iteration over the hyper-parameter space, varying one hyperparameter at a
         * time. Mutates the indices that are passed in and returns them.  Returns NULL when
         * the entire space has been traversed.
         */
        private int[] nextModelIndices(int[] hyperparamIndices) {
          // todo: Karthik, if grouped_parameters is in the _hyperParamNames, you need make sure not to use it
          // todo: in the hyperspace search.  However, you also need to make sure the correct parameter combinations
          // todo:  are chosen.  Make sure the right sizes arrays are chosen in the grouped_parameters
          // Find the next parm to flip
          int i;
          for (i = 0; i < hyperparamIndices.length; i++) {
            if (hyperparamIndices[i] + 1 < _hyperParams.get(_hyperParamNames[i]).length) {
              break;
            }
          }
          if (i == hyperparamIndices.length) {
            return null; // All done, report null
          }
          // Flip indices
          for (int j = 0; j < i; j++) {
            hyperparamIndices[j] = 0;
          }
          
          hyperparamIndices[i]++;
          if(_search_criteria._grouped_parameters != null) {
            int grouped_params_array_len = -1;
            Set<String> grouped_params = new HashSet<>(Arrays.asList(_search_criteria._grouped_parameters));
            for(int index = 0; index < hyperparamIndices.length; index++) {
              if(grouped_params.contains(_hyperParamNames[index])) {
                int param_index = hyperparamIndices[index];
                int param_arrlen = ((ArrayList) _hyperParams.get(_hyperParamNames[index])[param_index]).toArray().length;
                if(grouped_params_array_len != -1 && param_arrlen != grouped_params_array_len) {
                   return nextModelIndices(hyperparamIndices);
                } else {
                  grouped_params_array_len = param_arrlen;
                }
              }
            }
          }
          
          return hyperparamIndices;
        }
      }; // anonymous HyperSpaceIterator class
    } // iterator()

  } // class CartesianWalker

  /**
   * Hyperparameter space walker which visits random combinations of hyperparameters whose possible values are
   * given in explicit lists as they are with CartesianWalker.
   */
  class RandomDiscreteValueWalker<MP extends Model.Parameters>
          extends BaseWalker<MP, RandomDiscreteValueSearchCriteria> {

    private Random _random;
    private boolean _set_model_seed_from_search_seed;  // true if model parameter seed is set to default value and false otherwise

    public RandomDiscreteValueWalker(MP params,
                                     Map<String, Object[]> hyperParams,
                                     ModelParametersBuilderFactory<MP> paramsBuilderFactory,
                                     RandomDiscreteValueSearchCriteria search_criteria) {
      super(params, hyperParams, paramsBuilderFactory, search_criteria);

      // seed the models using the search seed if it is the only one specified
      long defaultSeed = _defaultParams._seed;
      long actualSeed = _params._seed;
      long gridSeed = search_criteria.seed();
      _set_model_seed_from_search_seed = defaultSeed == actualSeed && defaultSeed != gridSeed;
      _random = gridSeed == defaultSeed ? new Random() : new Random(gridSeed);
    }

    /** Based on the last model, the given array of ScoringInfo, and our stopping criteria should we stop early? */
    @Override
    public boolean stopEarly(Model model, ScoringInfo[] sk) {
      return ScoreKeeper.stopEarly(ScoringInfo.scoreKeepers(sk),
              search_criteria().stopping_rounds(),
              ScoreKeeper.ProblemType.forSupervised(model._output.isClassifier()),
              search_criteria().stopping_metric(),
              search_criteria().stopping_tolerance(), "grid's best", true);
    }

    @Override
    public HyperSpaceIterator<MP> iterator() {
      return new HyperSpaceIterator<MP>() {
        /** All visited hyper params permutations, including the current one. */
        private final List<int[]> _visitedPermutations = new ArrayList<>();
        private final Set<Integer> _visitedPermutationHashes = new LinkedHashSet<>(); // for fast dupe lookup

        /** Current hyper params permutation. */
        private int[] _currentHyperparamIndices = null;

        /** One-based count of the permutations we've visited, primarily used as an index into _visitedHyperparamIndices. */
        private int _currentPermutationNum = 0;

        // TODO: override into a common subclass:
        @Override
        public MP nextModelParameters(Model previousModel) {
          // NOTE: nextModel checks _visitedHyperparamIndices and does not return a duplicate set of indices.
          // NOTE: in RandomDiscreteValueWalker nextModelIndices() returns a new array each time, rather than
          // mutating the last one.
          _currentHyperparamIndices = nextModelIndices();

          if (_currentHyperparamIndices != null) {
            _visitedPermutations.add(_currentHyperparamIndices);
            _visitedPermutationHashes.add(integerHash(_currentHyperparamIndices));
            _currentPermutationNum++; // NOTE: 1-based counting

            // Fill array of hyper-values
            Object[] hypers = hypers(_currentHyperparamIndices);
            // Get clone of parameters
            MP commonModelParams = (MP) _params.clone();
            // Fill model parameters
            MP params = getModelParams(commonModelParams, hypers);

            // add max_runtime_secs in search criteria into params if applicable
            if (_search_criteria != null && _search_criteria.strategy() == Strategy.RandomDiscrete) {
              // ToDo: model seed setting will be different for parallel model building.
              // ToDo: This implementation only works for sequential model building.
              if (_set_model_seed_from_search_seed) {
                // set model seed = search_criteria.seed+(0, 1, 2,..., model number)
                params._seed = _search_criteria.seed() + (model_number++);
              }
            }
            return params;
          } else {
            throw new NoSuchElementException("No more elements to explore in hyper-space!");
          }
        }

        @Override
        public boolean hasNext(Model previousModel) {
          // Note: we compare _currentPermutationNum to max_models, because it counts successfully created models, but
          // we compare _visitedPermutationHashes.size() to _maxHyperSpaceSize because we want to stop when we have attempted each combo.
          //
          // _currentPermutationNum is 1-based
          return (_visitedPermutationHashes.size() < _maxHyperSpaceSize &&
                  (search_criteria().max_models() == 0 || _currentPermutationNum < search_criteria().max_models())
          );
        }

        @Override
        public void onModelFailure(Model failedModel, Consumer<Object[]> withFailedModelHyperParams) {
          // FIXME: when using parallel grid search, there's no good reason to think that the current hyperparam indices where the ones used for the failed model
          _currentPermutationNum--;
          withFailedModelHyperParams.accept(hypers(_currentHyperparamIndices));
        }

        /**
         * Random iteration over the hyper-parameter space.  Does not repeat
         * previously-visited combinations.  Returns NULL when we've hit the stopping
         * criteria.
         */
        private int[] nextModelIndices() {
          //todo: Karthik, you need to change this up if grouped_parameters are found in
          // todo: _hyperParameters.  If it is not found, everything will go as before.  You will
          // todo: always look for it at the beginning.  If found, do your magic.  If not, do as before.
          int[] hyperparamIndices =  new int[_hyperParamNames.length];
          
          Set<String> grouped_params = null;
          int grouped_params_length = -1;
          if(_search_criteria._grouped_parameters != null) {
            grouped_params = new HashSet<>(Arrays.asList(_search_criteria._grouped_parameters));  
          }
          
          do {
            grouped_params_length = -1;
            for (int i = 0; i < _hyperParamNames.length; i++) {
              if(grouped_params != null && grouped_params.contains(_hyperParamNames[i]) && grouped_params_length == -1) {
                int index = _random.nextInt(_hyperParams.get(_hyperParamNames[i]).length);
                grouped_params_length = ((ArrayList) _hyperParams.get(_hyperParamNames[i])[index]).toArray().length;
                hyperparamIndices[i] = index;
              } else if(grouped_params != null && grouped_params.contains(_hyperParamNames[i]) && grouped_params_length != -1) {
                hyperparamIndices[i] = nextGroupedParamIndex(grouped_params_length, _hyperParamNames[i], _hyperParams.get(_hyperParamNames[i]));
              } else {
                hyperparamIndices[i] = _random.nextInt(_hyperParams.get(_hyperParamNames[i]).length);  
              }
            }
            // check for aliases and loop if we've visited this combo before
          } while (_visitedPermutationHashes.contains(integerHash(hyperparamIndices)));

          Object[] a1 = ((ArrayList) _hyperParams.get(_hyperParamNames[0])[hyperparamIndices[0]]).toArray();
          Object[] a2 = ((ArrayList) _hyperParams.get(_hyperParamNames[1])[hyperparamIndices[1]]).toArray();
          Object[] a3 = ((ArrayList) _hyperParams.get(_hyperParamNames[2])[hyperparamIndices[2]]).toArray();
          return hyperparamIndices;
        } // nextModel

        /**
         * Returns random index in the hyperspace of the grouped
         * hyperparameter, param, subject to the constraint that the length of
         * the random array chosen is equal to grouped_params_length, which is
         * the length of the arrays of the other grouped parameters chosen.
         * 
         * @param grouped_params_length
         * @param param
         * @param param_values
         * @return
         */
        private int nextGroupedParamIndex(int grouped_params_length, String param, Object[] param_values) {
          int[] filtered_param_indices = IntStream.range(0, _hyperParams.get(param).length)
                  .filter(i -> ((ArrayList<Integer>)(param_values[i])).size() == grouped_params_length).toArray();
          return filtered_param_indices[_random.nextInt(filtered_param_indices.length)];
        }

      }; // anonymous HyperSpaceIterator class
    } // iterator()

  } // RandomDiscreteValueWalker
}
