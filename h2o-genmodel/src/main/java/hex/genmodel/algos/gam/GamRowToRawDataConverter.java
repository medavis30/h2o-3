package hex.genmodel.algos.gam;

import hex.genmodel.GenModel;
import hex.genmodel.easy.CategoricalEncoder;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.RowToRawDataConverter;
import hex.genmodel.easy.exception.PredictException;

import java.util.Map;

import static hex.genmodel.utils.ArrayUtils.nanArray;

public class GamRowToRawDataConverter extends RowToRawDataConverter {
  GamMojoModelBase _m;
  public GamRowToRawDataConverter(GenModel m, Map<String, Integer> modelColumnNameToIndexMap, Map<Integer, CategoricalEncoder> domainMap, EasyPredictModelWrapper.ErrorConsumer errorConsumer, EasyPredictModelWrapper.Config config) {
    super(m, modelColumnNameToIndexMap, domainMap, errorConsumer, config);
    _m = (GamMojoModelBase) m;
  }
  
  @Override
  public double[] convert(RowData data, double[] rawData) throws PredictException {
    if (_m.getFeautesPresentInData() == 0) 
      _m.countFeautesPresentInData(data.keySet()); // set _featuresPresentInData if it has not been set.
    if (rawData.length != _m.getFeautesPresentInData()) {
      rawData = nanArray(_m.getTotFeatureSize());
      rawData = super.convert(data, rawData);
      _m.addExpandGamCols(rawData, data);
      return rawData;
    } else 
      return super.convert(data, rawData);
  }
}
