// ============================================================================
//  MCAnalysis.C
//  Select pionton pairs (J/psi -> ee / mumu + backgrounds) from MC.
//  Per-sample output:
//    - ../Data/MLData<name>.root          : MLDataTree for machine learning
//    - Plots/Control/<name>_hists.root    : control histograms
// ============================================================================
#include <TROOT.h>
#include <TError.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TVector2.h>
#include <TLorentzVector.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <iostream>
#include <memory>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// Constants and selection criteria
// ----------------------------------------------------------------------------
constexpr double kPionMass     = 0.13957018;   // GeV
constexpr double kRad2Deg      = 180. / TMath::Pi();

constexpr double kMaxTrackEta = 2.5;
constexpr double kMinTrackPt  = 1.0;   // GeV
constexpr double kMaxD0       = 1.0;   // jednostka taka jak w gałęzi track_d0 (zwykle mm)
constexpr double kMaxPairPt   = 0.2;   // GeV
constexpr double kMaxPairMass = 3.5;   // GeV
constexpr double kMinPairMass = 2.0;   // GeV

// selekcja topoclusterów
constexpr double kMaxClusterEta = 4.9;
constexpr double kMinClusterPt  = 0.1; // GeV

constexpr double kMatchDeltaR     = 0.8;   // track <-> topocluster matching
constexpr double kMaxTruthMatchDR = 0.1;   // track <-> truth pion matching
constexpr int    kLogEvery        = 300000;

// ----------------------------------------------------------------------------
// Sample description
// ----------------------------------------------------------------------------
struct Sample {
    TString file;
    TString name;
    int    ParticleType;  
};

const std::vector<Sample> kSamples = {
    {"../Data/mc_various_particles.root","MCPionFull",  2},
};

// ----------------------------------------------------------------------------
// ML tree row
// ----------------------------------------------------------------------------
struct MLRow {
    Float_t fVariable = 0, emProb = 0, lambda = 0, lambda2 = 0, radius = 0;
    Float_t etaRange = 0;
    Int_t  ParticleType = -1;
};

void BookMLBranches(TTree& t, MLRow& r)
{
    t.Branch("Cal_FVariable",   &r.fVariable, "Cal_FVariable/F");
    t.Branch("Cal_EMprop",      &r.emProb,    "Cal_EMprop/F");
    t.Branch("Cal_Lambda",      &r.lambda,    "Cal_Lambda/F");
    t.Branch("Cal_Lambda2",     &r.lambda2,   "Cal_Lambda2/F");
    t.Branch("Cal_Radius",      &r.radius,    "Cal_Radius/F");
    t.Branch("EtaRange",        &r.etaRange,  "EtaRange/F");
    t.Branch("ParticleType",    &r.ParticleType, "ParticleType/I");
}

// ----------------------------------------------------------------------------
// Calorimeter variables for the matched topocluster
// (if no match is found: found=false, all variables are -1)
// ----------------------------------------------------------------------------
struct ClusterVars {
    bool    found    = false;
    int     nMatched = 0;      // number of clusters in the dR < kMatchDeltaR cone
    Float_t pt       = 0;      // pT of the selected (highest-pT) cluster
    Float_t f = -1, em = -1, lambda = -1, lambda2 = -1, radius = -1;
};

// ----------------------------------------------------------------------------
// Branch readers
// ----------------------------------------------------------------------------
struct TrackReaders {
    TTreeReaderArray<float> eta, phi, pt, pixeldEdX, pixelHits, sctHits, trtHits, d0;
    TTreeReaderArray<int>   charge;
    TTreeReaderArray<bool>  TriggerPass, Loose;

    explicit TrackReaders(TTreeReader& r)
        : eta(r, "track_eta"), phi(r, "track_phi"), pt(r, "track_pt"), 
          pixeldEdX(r, "track_PixeldEdX"), pixelHits(r, "track_PixelHits"),
          sctHits(r, "track_SCTHits"), trtHits(r, "track_TRTHits"), d0(r, "track_d0"),
          charge(r, "track_charge"),
          TriggerPass(r, "passed_HLT_mb_excl_1trk5_pt1_hi_FgapAC5_L1TRT_VTE20"),
          Loose(r, "track_is_LoosePrimary") {}
};

struct MCReaders {
    TTreeReaderArray<float> Eta, Phi, Pt;
    TTreeReaderArray<int>   PDG;

    explicit MCReaders(TTreeReader& r)
        : Pt(r, "truth_pion_pt"), Eta(r, "truth_pion_eta"), Phi(r, "truth_pion_phi"), PDG(r, "truth_pion_pdg") {}
};

// ----------------------------------------------------------------------------
// Histograms (one set per sample; each sample contains either electrons or muons)
// ----------------------------------------------------------------------------
struct Hists {
    // pion
    TH1D *pionE, *pionEta, *pionPt, *pionPhi, *MCcharge;
    // Track
    TH1D *pixelHits, *trtHits, *sctHits, *pixeldEdX, *trackCharge;
    // Topoclusters
    TH1D *nTopo, *perp, *fVar, *emProb, *lambda, *lambda2, *radius;
    // Other
    TH1D *etaRange , *hDR, *truthCount;
    TH1D *REtaPhiHist;

    TH2D *hDEtaDPhi;

    explicit Hists(const TString& prefix)
    {
        auto book = [&](const char* n, const char* title, int nb, double lo, double hi) {
            return new TH1D(prefix + "_" + n, title, nb, lo, hi);
        };
        const double pi = TMath::Pi();

        truthCount  = book("truthCount", "Truth pions per event;N_{truth #pi};Events", 11, -0.5, 10.5);
        REtaPhiHist = book("REtaPhiHist", "REtaPhiHist;#DeltaR(track, truth);Tracks", 100, 0, 8);
        MCcharge   = book("MCcharge",   "MC charge;Q;pions",    100, -1.5, 1.5);
        pionE   = book("pionE",   "pion energy;E [GeV];pions",    50, 0, 6);
        pionEta = book("pionEta", "pion #eta;#eta;pions",         50, -3, 3);
        pionPt  = book("pionPt",  "pion p_{T};p_{T} [GeV];pions", 50, 0, 1.8);
        pionPhi = book("pionPhi", "pion #phi;#phi;pions",         30, -pi, pi);
        pionPhi->SetMinimum(0);

        pixelHits = book("pixelHits", "Pixel hits;N_{pixel};Tracks", 11, -0.5, 10.5);
        trtHits   = book("trtHits",   "TRT hits;N_{TRT};Tracks",     51, -1, 50);
        trackCharge   = book("trackCharge",   "Track charge;Q;Tracks",     100, -1.5, 1.5);

        sctHits   = book("sctHits",   "SCT hits;N_{SCT};Tracks",     12, 2.5, 14.5);
        pixeldEdX = book("pixeldEdX", "Pixel dE/dx;dE/dx;Tracks",   100, 0, 2);

        nTopo   = book("nTopo",   "Number of topoclusters;N;pions",           15, -0.5, 14.5);
        perp    = book("perp",    "Cluster p_{T};p_{T};pions",              100, 0, 3.5);
        fVar    = book("fVar",    "F variable;F;pions",                     100, 0, 4);
        emProb  = book("emProb",  "EM calorimeter probability;P_{EM};pions", 40, -0.1, 1.1);
        lambda  = book("lambda",  "Cluster #lambda;#lambda;pions",           40, 0, 4);
        lambda2 = book("lambda2", "Cluster #lambda_{2};#lambda_{2};pions",   40, 0, 600);
        radius  = book("radius",  "Cluster radius;R;pions",                  40, 0, 100);
        
        hDR = book("dR", "#DeltaR(track, cluster), all clusters passing the cut;#DeltaR;Clusters", 120, 0, 6);
        hDEtaDPhi = new TH2D(prefix + "_dEta_dPhi", "Track - cluster;#Delta#eta;#Delta#phi;Clusters",
                               60, -kMatchDeltaR, kMatchDeltaR, 60, -kMatchDeltaR, kMatchDeltaR);

        etaRange = book("etaRange", "|#eta| range;range;pions", 3, 0.5, 3.5);
    }

    void FillCluster(const ClusterVars& c)
    {
        nTopo->Fill(c.nMatched);
        if (!c.found) return;

        perp->Fill(c.pt);
        fVar->Fill(c.f);
        emProb->Fill(c.em);
        lambda->Fill(c.lambda / 1000);
        lambda2->Fill(c.lambda2 / 1000);
        radius->Fill(c.radius / 1000);
    }
};

struct TopoReaders {
    TTreeReaderArray<float> eta, phi, pt, lambda, lambda2, r2, emProb;
    TTreeReaderArray<bool> pass;

    explicit TopoReaders(TTreeReader& r)
        : eta(r, "topo_cluster_eta"), phi(r, "topo_cluster_phi"),
          pt(r, "topo_cluster_pt"), lambda(r, "topo_cluster_lambda"),
          lambda2(r, "topo_cluster_lambda2"), r2(r, "topo_cluster_r2"),
          emProb(r, "topo_cluster_EM_prob"), pass(r, "topo_cluster_pass_sig_cut") {}

    // Finds clusters passing the signal selection within dR < kMatchDeltaR of
    // the pion and selects the one with the highest pT.
    ClusterVars Match(const TLorentzVector& pion, Hists& h) const
    {
        ClusterVars c;
        int best = -1;

        for (int i = 0; i < int(pt.GetSize()); ++i) {
            if (TMath::Abs(eta[i]) >= kMaxClusterEta || pt[i] <= kMinClusterPt) continue;

            const double dPhi = TVector2::Phi_mpi_pi(pion.Phi() - phi[i]);
            const double dEta = pion.Eta() - eta[i];
            const double dR   = TMath::Sqrt(dEta * dEta + dPhi * dPhi);
            h.hDEtaDPhi->Fill(dEta, dPhi);
            h.hDR->Fill(dR);

            if (dR >= kMatchDeltaR || !pass[i]) continue;

            ++c.nMatched;
            if (pt[i] > c.pt) { c.pt = pt[i]; best = i; }
        }

        if (best >= 0) {
            c.found   = true;
            c.f       = pt[best] / pion.Perp();
            c.em      = emProb[best];
            c.lambda  = lambda[best];
            c.lambda2 = lambda2[best];
            c.radius  = r2[best];
        }
        return c;
    }
};

// 1: |eta|<1, 2: 1<=|eta|<1.5, 3: remaining range (<2.5)
inline int EtaBin(double eta)
{
    const double a = TMath::Abs(eta);
    return a < 1.0 ? 1 : (a < 1.5 ? 2 : 3);
}

inline double DeltaR(double eta1, double phi1, double eta2, double phi2)
{
    const double dEta = eta1 - eta2;
    const double dPhi = TVector2::Phi_mpi_pi(phi1 - phi2);
    return TMath::Sqrt(dEta * dEta + dPhi * dPhi);
}

// ----------------------------------------------------------------------------
// Process one sample
// ----------------------------------------------------------------------------
void ProcessSample(const Sample& s)
{
    const double mass = kPionMass;

    // --- output: ML tree ---
    std::unique_ptr<TFile> mlFile(TFile::Open(Form("../Classifiers/MLData3Type%s.root", s.name.Data()), "RECREATE"));
    auto* mlTree = new TTree("MLDataTree", "MLDataTree");
    MLRow row;
    BookMLBranches(*mlTree, row);

    // --- output: histograms (created after cd(), so they are written to this file) ---
    std::unique_ptr<TFile> histFile(TFile::Open(Form("Plots/TrainingControl/%s_hists.root", s.name.Data()), "RECREATE"));
    histFile->cd();
    Hists h(s.name);

    // --- input ---
    TChain chain("G2TauTree");
    chain.Add(s.file);
    TTreeReader reader(&chain);
    TrackReaders trk(reader);
    MCReaders TruthPion(reader);
    TopoReaders  topo(reader);

    // For muons, reject pions without a matched cluster.
    // This selection was previously commented out for electrons, so it remains disabled.
    const bool requireCluster = true;

    long nEvents = 0, npions = 0, nTotalTruthPions = 0;

    while (reader.Next()) {
        if (++nEvents % kLogEvery == 0)
            std::cout << "Processing " << nEvents << " event..." << std::endl;

        // --- MC selection & count ---
        int nTruthPions = 0;

        for (int m = 0; m < int(TruthPion.Pt.GetSize()); ++m) {
            if (TruthPion.Pt[m] > 1.0) ++nTruthPions;
            //cout<<TruthPion.PDG[m]<<endl;
            h.MCcharge->Fill(TruthPion.PDG[m]/211);
        }
        h.truthCount->Fill(nTruthPions);
        nTotalTruthPions += nTruthPions;

        if (nTruthPions == 0) continue;

        // --- track selection & matching against all truth pions in event ---
        for (int i = 0; i < int(trk.pt.GetSize()); ++i) {
            if (TMath::Abs(trk.eta[i]) >= kMaxTrackEta || trk.pt[i] <= kMinTrackPt) continue;
            if (TMath::Abs(trk.d0[i]) >= kMaxD0 || !trk.Loose[i]) continue;

            double bestDR = 1e9;
            for (int m = 0; m < int(TruthPion.Pt.GetSize()); ++m) {
                if (TruthPion.Pt[m] <= 1.0) continue;
                const double dR = DeltaR(trk.eta[i], trk.phi[i], TruthPion.Eta[m], TruthPion.Phi[m]);
                h.REtaPhiHist->Fill(dR);
                if (dR < bestDR) bestDR = dR;
            }

            if (bestDR > kMaxTruthMatchDR) continue;

            TLorentzVector Pion4;
            Pion4.SetPtEtaPhiM(trk.pt[i], trk.eta[i], trk.phi[i], mass);

            h.pixeldEdX->Fill(trk.pixeldEdX[i]);
            h.pixelHits->Fill(trk.pixelHits[i]);
            h.sctHits->Fill(trk.sctHits[i]);
            h.trtHits->Fill(trk.trtHits[i]);
            h.trackCharge->Fill(trk.charge[i]);
            h.pionE->Fill(Pion4.E());
            h.pionEta->Fill(Pion4.Eta());
            h.pionPt->Fill(Pion4.Pt());
            h.pionPhi->Fill(Pion4.Phi());

            const ClusterVars c = topo.Match(Pion4, h);
            h.FillCluster(c);
            if (requireCluster && !c.found) continue;

            row.fVariable    = c.f;
            row.emProb       = c.em;
            row.lambda       = c.lambda;
            row.lambda2      = c.lambda2;
            row.radius       = c.radius;
            row.etaRange     = EtaBin(Pion4.Eta());
            row.ParticleType = s.ParticleType;

            mlTree->Fill();
            h.etaRange->Fill(row.etaRange);
            ++npions;
        }
    }

    std::cout << "===========================\n"
              << "End of file: " << s.name << "\n"
              << "Number of events: " << nEvents << "\n"
              << "Number of truth pions (pT > 1 GeV): " << nTotalTruthPions << "\n"
              << "Number of pions: " << npions << "\n"
              << "===========================" << std::endl;

    // --- save output ---
    histFile->cd();
    histFile->Write();
    histFile->Close();

    mlFile->cd();
    mlTree->Write();
    mlFile->Close();
}

} // namespace

// ----------------------------------------------------------------------------
void TrainingPions()
{
    gROOT->SetBatch(kTRUE);
    gErrorIgnoreLevel = kFatal;

    for (const auto& s : kSamples) ProcessSample(s);
}