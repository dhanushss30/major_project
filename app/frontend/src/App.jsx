import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
import SpeciesPage from './pages/SpeciesPage'
import SpeciesDetailPage from './pages/SpeciesDetailPage'
import ModelInfoPage from './pages/ModelInfoPage'
import ChatPage from './pages/ChatPage'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"                       element={<HomePage />} />
        <Route path="/predict"                element={<PredictPage />} />
        <Route path="/dashboard"              element={<DashboardPage />} />
        <Route path="/species"                element={<SpeciesPage />} />
        <Route path="/species/:code"          element={<SpeciesDetailPage />} />
        <Route path="/model"                  element={<ModelInfoPage />} />
        <Route path="/chat"                   element={<ChatPage />} />
      </Routes>
    </Layout>
  )
}
