// ============================================================================
//  ShapePlotter.C  -  shape comparison of the classifier input features
//
//  Run:
//      root -l -b -q 'ShapePlotter.C+'
//
//  One page per feature, all samples on ONE plot (normalised to unit area):
//      Data electrons, Data muons, MC J/psi->ee, MC J/psi->mumu, pions
//
//  Input files = output of TestingMacro.C (Histograms<name>.root).
//  Output: <outDir>/ShapeComparison.pdf (multi-page, with bookmarks)
// ============================================================================

#include <TAxis.h>
#include <TCanvas.h>
#include <TError.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPad.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace shp {

// ----------------------------------------------------------------------------
//  Configuration  (EDIT PATHS HERE)
// ----------------------------------------------------------------------------
const std::string kDir      = "Plots/TestingControl/";
const std::string kFileData = kDir + "HistogramsTrueData.root";
const std::string kFileMCee = kDir + "HistogramsMC_ee.root";
const std::string kFileMCmm = kDir + "HistogramsMC_mumu.root";
const std::string kFilePion = kDir + "Histograms_SS_TrueData.root";

// Where the cluster histograms live in the pion file.
// Default: "clusters_all/" without prefix (all tracks, no e/mu classification).
// If the pion file was made with the e/mu split, use e.g. "electron/" + "clu_".
const std::string kPionDir    = "clusters_all/";
const std::string kPionPrefix = "";

const Color_t kColData   = kBlack;
const Color_t kColDataMu = kBlack;
const Color_t kColEl     = kRed + 1;
const Color_t kColMu     = kAzure + 2;
const Color_t kColPion   = kGreen + 2;
const char*   kLabel     = "J/#psi #rightarrow l^{+}l^{-} selection";

// ----------------------------------------------------------------------------
//  Structures
// ----------------------------------------------------------------------------
struct Feature {            // classifier input
   const char* name;        // histogram name without prefix
   const char* xTitle;
   bool        logY;
};

struct Entry {              // one curve in the plot
   TFile*      file;
   std::string path;        // directory + prefix, e.g. "electron/clu_"
   std::string label;
   Color_t     col;
   int         marker;
   bool        isData;      // data: points only; otherwise: filled histogram + points
};

// Order = order of the classifier inputs {f, emProb, lambda, lambda2, r2}
const std::vector<Feature> kFeatures = {
   {"f",       "F = p_{T}^{cl} / p_{T}^{trk}", false},
   {"emProb",  "Cluster EM probability",       true},
   {"lambda",  "Cluster #lambda / 10^{3}",     false},
   {"lambda2", "Cluster #lambda^{2} / 10^{3}", true},
   {"radius",  "Cluster radius^{2} / 10^{3}",  true},
};

// ----------------------------------------------------------------------------
//  Utilities
// ----------------------------------------------------------------------------
std::string Uniq()
{
   static int n = 0;
   return Form("shp_%d", n++);
}

TH1D* Get(TFile* f, const std::string& path)
{
   auto* h = dynamic_cast<TH1D*>(f->Get(path.c_str()));
   if (!h) std::cerr << "Histogram not found: " << f->GetName() << " : " << path << std::endl;
   return h;
}

void SetStyle()
{
   gROOT->SetBatch(kTRUE);
   gErrorIgnoreLevel = kWarning;
   TH1::AddDirectory(kFALSE);

   gStyle->SetOptStat(0);
   gStyle->SetOptTitle(0);
   gStyle->SetEndErrorSize(0);

   gStyle->SetTextFont(42);
   gStyle->SetLabelFont(42, "XYZ");
   gStyle->SetTitleFont(42, "XYZ");
   gStyle->SetLabelSize(0.045, "XYZ");
   gStyle->SetTitleSize(0.05, "XYZ");
   gStyle->SetTitleOffset(1.25, "X");
   gStyle->SetTitleOffset(1.4, "Y");

   gStyle->SetPadLeftMargin(0.14);
   gStyle->SetPadRightMargin(0.05);
   gStyle->SetPadBottomMargin(0.13);
   gStyle->SetPadTopMargin(0.06);
   gStyle->SetPadTickX(1);
   gStyle->SetPadTickY(1);
   gStyle->SetLegendBorderSize(0);
   gStyle->SetLegendFillStyle(0);
   gStyle->SetLegendFont(42);
}

std::vector<std::pair<TCanvas*, std::string>>& Pages()
{
   static std::vector<std::pair<TCanvas*, std::string>> pages;
   return pages;
}

void WritePDF(const std::string& path)
{
   auto& p = Pages();
   if (p.empty()) { std::cerr << "No plots to save" << std::endl; return; }
   const size_t n = p.size();
   for (size_t i = 0; i < n; ++i) {
      const char* suffix = (n == 1) ? "" : (i == 0 ? "(" : (i == n - 1 ? ")" : ""));
      p[i].first->Print((path + suffix).c_str(), ("Title:" + p[i].second).c_str());
      delete p[i].first;
   }
   p.clear();
}

void Note(const std::string& s, double x, double y, double size = 0.042)
{
   TLatex t;
   t.SetNDC();
   t.SetTextFont(42);
   t.SetTextSize(size);
   t.DrawLatex(x, y, s.c_str());
}

double MinPositive(const TH1* h)
{
   double m = 1e30;
   for (int i = 1; i <= h->GetNbinsX(); ++i) {
      const double v = h->GetBinContent(i);
      if (v > 0 && v < m) m = v;
   }
   return m;
}

// ----------------------------------------------------------------------------
//  One plot: all entries overlaid on a single canvas
// ----------------------------------------------------------------------------
void Compare(const Feature& v, const std::vector<Entry>& entries)
{
   std::vector<TH1D*>  hs;
   std::vector<double> nEntries;

   for (const auto& e : entries) {
      TH1D* h0 = Get(e.file, e.path + v.name);
      if (!h0) return;
      if (h0->Integral() <= 0) {
         std::cerr << "Empty histogram: " << e.path + v.name << " (" << e.label << ")" << std::endl;
         return;
      }
      if (!hs.empty() && h0->GetNbinsX() != hs[0]->GetNbinsX()) {
         std::cerr << "Binning mismatch: " << e.path + v.name << std::endl;
         return;
      }
      nEntries.push_back(h0->GetEntries());
      auto* h = static_cast<TH1D*>(h0->Clone(Uniq().c_str()));
      h->Sumw2();
      hs.push_back(h);
   }

   // Normalise to unit area
   for (auto* h : hs) h->Scale(1. / h->Integral());

   // Style
   for (size_t i = 0; i < hs.size(); ++i) {
      hs[i]->SetTitle("");
      hs[i]->SetLineColor(entries[i].col);
      hs[i]->SetMarkerColor(entries[i].col);
      hs[i]->SetMarkerStyle(entries[i].marker);
      hs[i]->SetMarkerSize(0.9);
      hs[i]->SetLineWidth(2);
      if (entries[i].isData) hs[i]->SetFillStyle(0);
      else                   hs[i]->SetFillColorAlpha(entries[i].col, 0.12);
      hs[i]->GetXaxis()->SetTitle(v.xTitle);
      hs[i]->GetYaxis()->SetTitle("Normalised entries");
   }

   // Y range
   double ymaxAll = 0, minPos = 1e30;
   for (auto* h : hs) {
      ymaxAll = std::max(ymaxAll, h->GetMaximum());
      minPos  = std::min(minPos, MinPositive(h));
   }
   double ymin, ymax;
   if (v.logY) { ymin = minPos * 0.5; ymax = ymaxAll * 30; }
   else        { ymin = 0;            ymax = ymaxAll * 1.4; }

   // Canvas
   auto* c = new TCanvas(Uniq().c_str(), "", 800, 800);
   if (v.logY) c->SetLogy();

   // Frame from the first histogram, MC / pions as filled histograms, data points on top
   hs[0]->SetMinimum(ymin);
   hs[0]->SetMaximum(ymax);
   hs[0]->Draw("AXIS");
   for (size_t i = 0; i < hs.size(); ++i) {
      if (entries[i].isData) continue;
      hs[i]->Draw("HIST SAME");
      hs[i]->Draw("E1 SAME");
   }
   for (size_t i = 0; i < hs.size(); ++i)
      if (entries[i].isData) hs[i]->Draw("E1 SAME");

   auto* leg = new TLegend(0.17, 0.77, 0.60, 0.87);
   leg->SetNColumns(1);
   leg->SetTextSize(0.018);
   for (size_t i = 0; i < hs.size(); ++i)
      leg->AddEntry(hs[i], Form("%s (N = %.0f)", entries[i].label.c_str(), nEntries[i]),
                    entries[i].isData ? "lep" : "lf");
   leg->Draw();
   Note(kLabel, 0.17, 0.90);

   Pages().emplace_back(c, v.name);
}

}  // namespace shp

// ============================================================================
//  Main function
// ============================================================================
void ShapePlotter()
{
   using namespace shp;
   SetStyle();

   TFile* fData = TFile::Open(kFileData.c_str(), "READ");
   TFile* fMCee = TFile::Open(kFileMCee.c_str(), "READ");
   TFile* fMCmm = TFile::Open(kFileMCmm.c_str(), "READ");
   TFile* fPion = TFile::Open(kFilePion.c_str(), "READ");
   for (TFile* f : {fData, fMCee, fMCmm, fPion}) {
      if (!f || f->IsZombie()) {
         std::cerr << "Cannot open one of the input files (check paths at the top of the macro)" << std::endl;
         return;
      }
   }

   // Curves on every plot (the first one sets the frame)
   const std::vector<Entry> entries = {
      {fData, "electron/clu_", "Data electrons",                kColData,   20, true},
      {fData, "muon/clu_",     "Data muons",                    kColDataMu, 24, true},
      {fMCee, "electron/clu_", "MC J/#psi #rightarrow ee",      kColEl,     21, false},
      {fMCmm, "muon/clu_",     "MC J/#psi #rightarrow #mu#mu",  kColMu,     22, false},
      {fPion, kPionDir + kPionPrefix, "Pions",                  kColPion,   23, false},
   };

   for (const auto& v : kFeatures) Compare(v, entries);

   gSystem->mkdir(kDir.c_str(), kTRUE);
   WritePDF(kDir + "ShapeComparison.pdf");

   for (TFile* f : {fData, fMCee, fMCmm, fPion}) f->Close();
}