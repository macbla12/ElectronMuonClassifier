// ============================================================================
//  RFClassifier.cpp  -  e/mu Random Forest classifier (ONNX Runtime)
//
//  WYMAGANIA
//    1) ONNX Runtime (C++ API) - pobrac gotowa paczke z:
//         https://github.com/microsoft/onnxruntime/releases
//       (np. onnxruntime-linux-x64-1.x.y.tgz) i rozpakowac, np. do ~/onnxruntime
//    2) Plik modelu:  RFMuonElectron.onnx
//  UZYCIE W WLASNYM KODZIE / ROOT-owym makrze
//
//       #include "RFClassifier.cpp"
//       RFClassifier clf("RFMuonElectron.onnx");
//       float pMuon = clf.Predict(f, emProb, lambda, lambda2, r2);
//
//  INPUT  (5 liczb float, DOKLADNIE w tej kolejnosci, dla JEDNEGO tracka
//          z dopasowanym topoclusterem):
//          [0] f        = pT_cluster / pT_track
//          [1] emProb   = topo_cluster_EM_prob
//          [2] lambda   = topo_cluster_lambda
//          [3] lambda2  = topo_cluster_lambda2
//          [4] r2       = topo_cluster_r2
//
//  OUTPUT (float):
//       P(muon) w zakresie [0, 1]
//       bliskie 0 -> elektron,  bliskie 1 -> mion
// ============================================================================

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

class RFClassifier {
public:
   // path - sciezka do pliku .onnx
   explicit RFClassifier(const char* path) : fEnv(ORT_LOGGING_LEVEL_WARNING, "rf"), fSession(nullptr)
   {
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(1);
      fSession = Ort::Session(fEnv, path, opts);

      Ort::AllocatorWithDefaultOptions alloc;
      fInputName = fSession.GetInputNameAllocated(0, alloc).get();
   }

   // Zwraca P(muon). x = {f, emProb, lambda, lambda2, r2}
   float Predict(std::array<float, 5>& x)
   {
      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::array<int64_t, 2> shape{1, 5};
      Ort::Value input = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), shape.size());

      const char* inNames[]  = {fInputName.c_str()};
      const char* outNames[] = {"label", "probabilities"};
      auto out = fSession.Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 2);
      return out[1].GetTensorMutableData<float>()[1];
   }

   // Wygodny przeciazony wariant
   float Predict(float f, float emProb, float lambda, float lambda2, float r2)
   {
      std::array<float, 5> x = {f, emProb, lambda, lambda2, r2};
      return Predict(x);
   }

private:
   Ort::Env     fEnv;
   Ort::Session fSession;
   std::string  fInputName;
};
